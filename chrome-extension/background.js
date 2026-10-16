// Only tab IDs created by this extension can be used for browser actions.
const owned = new Set();
const owners = new Map();
let socket;
let lastAttempt = -Infinity;
let retryTimer;
let retriesLeft = 2;
let enabled = false;
const ready = Promise.all([
  chrome.storage.local.get('enabled'),
  chrome.storage.session.get(['ownedTabs', 'tabOwners']),
]).then(async ([preferences, {ownedTabs = [], tabOwners = {}}]) => {
  enabled = preferences.enabled === true;

  for (const id of ownedTabs) {
    try {
      await chrome.tabs.get(id);
      owned.add(id);
      if (tabOwners[id]) owners.set(id, tabOwners[id]);
    } catch { /* closed while worker slept */ }
  }
  await persistOwned();
});
function persistOwned() {
  return chrome.storage.session.set({ownedTabs: [...owned], tabOwners: Object.fromEntries(owners)});
}

chrome.alarms.create('bridge-check', {periodInMinutes: 0.5});
chrome.alarms.onAlarm.addListener(alarm => {
  if (alarm.name === 'bridge-check') ready.then(() => {
    retriesLeft = 2;
    connect();
  });
});
chrome.tabs.onRemoved.addListener(async id => {
  await ready;
  owned.delete(id);
  owners.delete(id);
  await persistOwned();
});
chrome.runtime.onMessage.addListener((message, _sender, respond) => {
  if (message.action === 'status') {
    ready.then(() => {
      connect();
      respond({enabled, connected: enabled && socket?.readyState === WebSocket.OPEN && socket.authenticated === true});
    });
    return true;
  }
  if (message.action === 'setEnabled' && typeof message.enabled === 'boolean') {
    ready.then(async () => {
      // Flip the in-memory gate before persisting so in-flight requests see off immediately.
      enabled = message.enabled;
      await chrome.storage.local.set({enabled});
      connect();
      sendStatus();
      respond({enabled});
    });
    return true;
  }
});

function sendStatus() {
  if (socket?.readyState === WebSocket.OPEN && socket.authenticated) {
    socket.send(JSON.stringify({event: 'status', enabled}));
  }
}

function connect() {
  if (socket && socket.readyState < WebSocket.CLOSING) return;
  if (Date.now() - lastAttempt < 2000) return;
  lastAttempt = Date.now();
  const ws = new WebSocket('ws://127.0.0.1:17777');
  socket = ws;
  ws.authenticated = false;
  ws.addEventListener('message', async event => {
    if (socket !== ws) return;
    const request = JSON.parse(event.data);
    if (request.connected) {
      ws.authenticated = true;
      retriesLeft = 2;
      sendStatus();
      return;
    }
    let reply;
    try { await ready; reply = {result: await handle(request)}; }
    catch (error) { reply = {error: String(error)}; }
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(reply));
  });
  ws.addEventListener('close', () => {
    if (socket !== ws) return;
    socket = undefined;
    if (retryTimer) clearTimeout(retryTimer);
    if (retriesLeft > 0) {
      retriesLeft--;
      retryTimer = setTimeout(() => { retryTimer = undefined; connect(); }, 2000);
    }
    // After the brief burst, the alarm retries on a relaxed cadence.
  });
  ws.addEventListener('error', () => ws.close());
}

async function groupAgentTab(tab) {
  // Reuse a group only through an owned tab in this window, never by scanning
  // pre-existing tabs or adopting somebody else's group by its title.
  let groupId;
  for (const id of owned) {
    if (id === tab.id) continue;
    try {
      const other = await chrome.tabs.get(id);
      if (other.windowId !== tab.windowId || other.groupId === chrome.tabGroups.TAB_GROUP_ID_NONE) continue;
      const group = await chrome.tabGroups.get(other.groupId);
      if (group.title === 'Agent' && group.color === 'green') {
        groupId = group.id;
        break;
      }
    } catch { /* The other owned tab or its group was closed. */ }
  }
  const id = await chrome.tabs.group(groupId === undefined ? {tabIds: [tab.id]} : {tabIds: [tab.id], groupId});
  await chrome.tabGroups.update(id, {title: 'Agent', color: 'green'});
}

function safeUrl(url) {
  const parsed = new URL(url);
  if (!['http:', 'https:'].includes(parsed.protocol)) throw Error('Only HTTP(S) URLs are allowed');
  return parsed.href;
}

async function handle(request) {
  const {action, tab_id: id} = request;
  if (action === 'close_owner') {
    const ids = [...owners].filter(([, owner]) => owner === request.owner).map(([tabId]) => tabId);
    for (const tabId of ids) {
      owned.delete(tabId);
      owners.delete(tabId);
      try { await chrome.tabs.remove(tabId); } catch { /* user closed tab first */ }
    }
    await persistOwned();
    return {closed: ids.length};
  }
  if (!enabled) throw Error('Agent Live Tabs is disabled');
  if (action === 'open') {
    const url = safeUrl(request.url);
    const tab = await chrome.tabs.create({url, active: false});
    owned.add(tab.id);
    if (request.owner) owners.set(tab.id, request.owner);
    await persistOwned();
    await groupAgentTab(tab);
    return {tab_id: tab.id, url};
  }
  if (!owned.has(id)) throw Error('Tab not opened by Agent; access denied');
  const tab = await chrome.tabs.get(id);
  if (action === 'close') {
    owned.delete(id);
    owners.delete(id);
    await persistOwned();
    await chrome.tabs.remove(id);
    return {closed: id};
  }
  if (action === 'navigate') {
    const url = safeUrl(request.url);
    await chrome.tabs.update(id, {url});
    return {tab_id: id, url};
  }
  if (!/^https?:\/\//.test(tab.url || '')) throw Error('Only HTTP(S) pages can be accessed');
  if (!['snapshot', 'click', 'type', 'scroll'].includes(action)) throw Error('Unknown action');
  const [{result}] = await chrome.scripting.executeScript({
    target: {tabId: id},
    func: pageAction,
    args: [action, request.selector, request.text, request.pixels],
  });
  if (result.error) throw Error(result.error);
  if (action === 'snapshot') {
    const current = await chrome.tabs.get(id);
    const group = current.groupId === chrome.tabGroups.TAB_GROUP_ID_NONE
      ? null : await chrome.tabGroups.get(current.groupId);
    return {tab_id: id, url: current.url, group: group && {id: group.id, title: group.title, color: group.color}, ...result};
  }
  return {tab_id: id, url: tab.url, ...result};
}

// Runs in the page's isolated world; never accepts arbitrary JavaScript from the agent.
function pageAction(action, selector, text, pixels) {
  try {
    if (action === 'snapshot') return {title: document.title, text: document.body?.innerText.slice(0, 20000) || '', links: [...document.querySelectorAll('a[href]')].slice(0, 100).map(a => ({text: a.innerText.slice(0, 120), href: a.href}))};
    if (action === 'scroll') { window.scrollBy(0, pixels ?? 600); return {scrolled: true}; }
    const el = document.querySelector(selector);
    if (!el) throw Error(`No element matching ${selector}`);
    if (action === 'click') { el.click(); return {clicked: true}; }
    if (action === 'type') {
      if (!('value' in el)) throw Error('Element is not a form input');
      el.focus();
      const prototype = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
      if (setter) setter.call(el, text); else el.value = text;
      el.dispatchEvent(new Event('input', {bubbles: true}));
      el.dispatchEvent(new Event('change', {bubbles: true}));
      return {typed: true};
    }
  } catch (error) { return {error: String(error)}; }
}

ready.then(connect);

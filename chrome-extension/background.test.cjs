const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function harness(savedEnabled = false) {
  let listener;
  let alarmListener;
  const local = {enabled: savedEnabled};
  const session = {ownedTabs: []};
  const tabs = new Map([[3, {id: 3, url: 'https://private.example', windowId: 1, groupId: -1}]]);
  const createCalls = [];
  const windowFocusCalls = [];
  const sockets = [];
  const timers = [];
  let now = 0;
  const groups = new Map();
  let nextGroupId = 10;
  let nextId = 4;
  class Socket {
    static OPEN = 1;
    static CLOSING = 2;
    constructor() { this.readyState = 1; this.authenticated = false; this.handlers = {}; sockets.push(this); }
    addEventListener(name, fn) { this.handlers[name] = fn; }
    send() {}
    close() { this.readyState = 3; this.handlers.close?.(); }
    receive(message) { this.handlers.message?.({data: JSON.stringify(message)}); }
  }
  const chrome = {
    storage: {
      local: {get: async () => ({...local}), set: async v => Object.assign(local, v)},
      session: {get: async () => ({...session}), set: async v => Object.assign(session, v)},
    },
    tabs: {
      onRemoved: {addListener() {}},
      async get(id) { if (!tabs.has(id)) throw Error('missing tab'); return tabs.get(id); },
      async create(options) {
        createCalls.push(options);
        const tab = {id: nextId++, url: options.url, windowId: 1, groupId: -1};
        tabs.set(tab.id, tab);
        return tab;
      },
      async group({tabIds, groupId}) {
        const id = groupId ?? nextGroupId++;
        const group = groups.get(id) ?? {id, windowId: tabs.get(tabIds[0]).windowId, title: '', color: 'grey'};
        groups.set(id, group);
        for (const tabId of tabIds) tabs.get(tabId).groupId = id;
        return id;
      },
      async remove(id) { tabs.delete(id); },
      async update(id, {url}) { tabs.get(id).url = url; },
    },
    windows: {async update(id, options) { windowFocusCalls.push({id, ...options}); }},
    tabGroups: {
      TAB_GROUP_ID_NONE: -1,
      async get(id) { if (!groups.has(id)) throw Error('missing group'); return groups.get(id); },
      async update(id, props) { Object.assign(groups.get(id), props); return groups.get(id); },
    },
    scripting: {async executeScript() { return [{result: {title: 'example'}}]; }},
    alarms: {create() {}, onAlarm: {addListener(fn) { alarmListener = fn; }}}, 
    runtime: {onMessage: {addListener(fn) { listener = fn; }}},
  };
  const context = vm.createContext({chrome, WebSocket: Socket, URL, Date: {now: () => now}, setTimeout: (fn, delay) => { timers.push({fn, delay}); return timers.length; }, clearTimeout() {}, console});
  vm.runInContext(fs.readFileSync(__dirname + '/background.js', 'utf8') + '\nthis.testHandle = handle; this.testReady = ready;', context);
  const send = message => new Promise(resolve => listener(message, {}, resolve));
  return {context, send, sockets, local, tabs, createCalls, windowFocusCalls, groups, timers, alarm: () => alarmListener({name: 'bridge-check'}), advance: ms => { now += ms; }};
}

test('disabled by default; enable connects, disable closes and blocks all commands', async () => {
  const h = harness();
  await h.context.testReady;
  await Promise.resolve();
  assert.equal(h.sockets.length, 1);
  assert.equal((await h.send({action: 'status'})).enabled, false);
  await assert.rejects(h.context.testHandle({action: 'open', url: 'https://example.org'}), /disabled/i);
  await h.send({action: 'setEnabled', enabled: true});
  assert.equal(h.sockets.length, 1);
  const opened = await h.context.testHandle({action: 'open', url: 'https://example.org'});
  assert.equal(opened.tab_id, 4);
  await assert.rejects(h.context.testHandle({action: 'snapshot', tab_id: 3}), /access denied/i);
  await h.send({action: 'setEnabled', enabled: false});
  assert.equal(h.sockets[0].readyState, 1);
  assert.equal(h.local.enabled, false);
  await assert.rejects(h.context.testHandle({action: 'snapshot', tab_id: 4}), /disabled/i);
  await assert.rejects(h.context.testHandle({action: 'open', url: 'https://example.org'}), /disabled/i);
  assert.equal(h.sockets.length, 1);
});

test('open creates a background tab without selecting it', async () => {
  const h = harness(true);
  await h.context.testReady;
  await h.context.testHandle({action: 'open', url: 'https://example.org'});
  assert.equal(h.createCalls.length, 1);
  assert.equal(h.createCalls[0].active, false);
  assert.equal(h.windowFocusCalls.length, 0);
});

test('open with focus selects the tab and brings its window forward', async () => {
  const h = harness(true);
  await h.context.testReady;
  const opened = await h.context.testHandle({action: 'open', url: 'https://example.org', focus: true});
  assert.equal(h.createCalls.length, 1);
  assert.equal(h.createCalls[0].active, true);
  assert.deepEqual(h.windowFocusCalls, [{id: h.tabs.get(opened.tab_id).windowId, focused: true}]);
});

test('new tabs join a green Agent group without grouping pre-existing tabs', async () => {
  const h = harness(true);
  await h.context.testReady;
  const first = await h.context.testHandle({action: 'open', url: 'https://example.org'});
  const second = await h.context.testHandle({action: 'open', url: 'https://www.wikipedia.org/'});
  assert.equal(h.tabs.get(3).groupId, -1);
  assert.notEqual(h.tabs.get(first.tab_id).groupId, -1);
  assert.equal(h.tabs.get(first.tab_id).groupId, h.tabs.get(second.tab_id).groupId);
  assert.equal(h.groups.size, 1);
  assert.equal(h.groups.get(h.tabs.get(first.tab_id).groupId).title, 'Agent');
  assert.equal(h.groups.get(h.tabs.get(first.tab_id).groupId).color, 'green');
  const snapshot = await h.context.testHandle({action: 'snapshot', tab_id: first.tab_id});
  assert.equal(snapshot.group.title, 'Agent');
  assert.equal(snapshot.group.color, 'green');
  assert.equal(snapshot.group.id, h.tabs.get(first.tab_id).groupId);
});

test('closing one agent session closes only its tabs', async () => {
  const h = harness(true);
  await h.context.testReady;
  const one = await h.context.testHandle({action: 'open', url: 'https://example.org', owner: 'agent-one'});
  const two = await h.context.testHandle({action: 'open', url: 'https://example.org', owner: 'agent-two'});
  await h.send({action: 'setEnabled', enabled: false});
  await h.context.testHandle({action: 'close_owner', owner: 'agent-one'});
  assert.equal(h.tabs.has(one.tab_id), false);
  assert.equal(h.tabs.has(two.tab_id), true);
  assert.equal(h.tabs.has(3), true);
  await h.send({action: 'setEnabled', enabled: true});
  await assert.rejects(h.context.testHandle({action: 'snapshot', tab_id: one.tab_id}), /access denied/);
});

test('off switch keeps status connection and refuses action frames', async () => {
  const h = harness(false);
  await h.context.testReady;
  await Promise.resolve();
  assert.equal(h.sockets.length, 1);
  assert.equal((await h.send({action: 'status'})).enabled, false);
  h.sockets[0].receive({connected: true});
  h.sockets[0].receive({action: 'open', url: 'https://example.org'});
  await Promise.resolve();
  assert.equal(h.tabs.size, 1);
  assert.equal(h.sockets[0].readyState, 1);
});

test('status request reconnects after failed bridge and reports current state', async () => {
  const h = harness(true);
  await h.context.testReady;
  await Promise.resolve();
  h.sockets[0].close();
  h.advance(2100);
  const state = await h.send({action: 'status'});
  assert.equal(state.connected, false);
  assert.equal(h.sockets.length, 2, 'checking status should retry a missing bridge');
  h.sockets[1].close();
  await h.send({action: 'status'});
  assert.equal(h.sockets.length, 2, 'repeat status checks respect reconnect cooldown');
});

test('failed connection gets a short bounded retry burst then falls back to alarm', async () => {
  const h = harness(true);
  await h.context.testReady;
  await Promise.resolve();
  h.sockets[0].close();
  assert.equal(h.timers.length, 1);
  assert.equal(h.timers[0].delay, 2000);
  h.advance(2000);
  h.timers.shift().fn();
  assert.equal(h.sockets.length, 2);
  h.sockets[1].close();
  assert.equal(h.timers.length, 1);
  h.advance(2000);
  h.timers.shift().fn();
  assert.equal(h.sockets.length, 3);
  h.sockets[2].close();
  assert.equal(h.timers.length, 0, 'no unbounded error loop');
});

test('unavailable bridge uses alarm retry instead of rapid error loop', async () => {
  const h = harness(true);
  await h.context.testReady;
  await Promise.resolve();
  assert.equal(h.sockets.length, 1);
  h.sockets[0].close(); // Simulate Chrome's connection-refused close.
  assert.equal(h.timers.length, 1, 'one bounded retry is allowed');
  h.advance(30000);
  h.alarm();
  await Promise.resolve();
  assert.equal(h.sockets.length, 2, 'alarm eventually retries');
});

test('saved switch state controls connection on service worker restart', async () => {
  const off = harness(false);
  await off.context.testReady;
  await Promise.resolve();
  assert.equal(off.sockets.length, 1);
  assert.equal((await off.send({action: 'status'})).enabled, false);
  const on = harness(true);
  await on.context.testReady;
  await Promise.resolve();
  assert.equal(on.sockets.length, 1);
  await on.send({action: 'setEnabled', enabled: false});
  on.sockets[0].receive({action: 'open', url: 'https://example.org'});
  await Promise.resolve();
  assert.equal(on.tabs.size, 1, 'stale socket cannot open tabs after switch off');
});

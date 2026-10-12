const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(__dirname + '/popup.html', 'utf8');
const css = fs.readFileSync(__dirname + '/popup.css', 'utf8');

test('popup has a dark green theme and a keyboard-accessible slider', () => {
  assert.match(html, /<link rel="stylesheet" href="popup\.css">/);
  assert.match(html, /<input[^>]*type="checkbox"[^>]*id="enabled"/);
  assert.match(html, /<span class="switch-track" aria-hidden="true">/);
  assert.match(html, /<span class="switch-thumb"><\/span>/);
  assert.match(css, /color-scheme:\s*dark/);
  assert.match(css, /#enabled:checked\s*\+\s*\.switch-track/);
  assert.match(css, /#enabled:focus-visible\s*\+\s*\.switch-track/);
  assert.doesNotMatch(css, /display:\s*none/);
});

test('slider preserves enabled and connection statuses', async () => {
  let enabled = false;
  let change;
  const elements = {
    state: {textContent: ''},
    enabled: {checked: false, disabled: false, addEventListener(name, fn) { if (name === 'change') change = fn; }},
  };
  const chrome = {runtime: {async sendMessage(message) {
    if (message.action === 'setEnabled') enabled = message.enabled;
    return {enabled, connected: enabled};
  }}};
  vm.runInNewContext(fs.readFileSync(__dirname + '/popup.js', 'utf8'), {
    chrome, document: {body: {dataset: {}}, getElementById: id => elements[id]},
  });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(elements.state.textContent, 'Off');
  elements.enabled.checked = true;
  await change();
  assert.equal(elements.state.textContent, 'Connected to Agent');
  elements.enabled.checked = false;
  await change();
  assert.equal(elements.state.textContent, 'Off');
});

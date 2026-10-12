const state = document.getElementById('state');
const toggle = document.getElementById('enabled');
function render({enabled, connected}) {
  toggle.checked = enabled;
  document.body.dataset.status = !enabled ? 'off' : connected ? 'connected' : 'waiting';
  state.textContent = !enabled ? 'Off' : connected ? 'Connected to Agent' : 'On — waiting for Agent to start browser tools';
}
async function refresh() {
  render(await chrome.runtime.sendMessage({action: 'status'}));
}
toggle.addEventListener('change', async () => {
  toggle.disabled = true;
  await chrome.runtime.sendMessage({action: 'setEnabled', enabled: toggle.checked});
  await refresh();
  toggle.disabled = false;
});
refresh();

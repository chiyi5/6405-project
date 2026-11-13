/* Translate UI text nodes without rebuilding forms or altering source documents. */
(() => {
  'use strict';
  const pairs = window.SKILLBRIDGE_TRANSLATIONS;
  const direct = new Map();
  const patterns = [];
  const escapeRegex = text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  for (const [en, zh] of pairs) {
    if (!en.includes('{')) {
      direct.set(en, {en, zh});
      direct.set(zh, {en, zh});
      continue;
    }
    for (const input of [en, zh]) {
      const keys = [];
      let previous = 0, expression = '^';
      for (const item of input.matchAll(/\{(\w+)\}/g)) {
        expression += escapeRegex(input.slice(previous, item.index)) + '(.+?)';
        keys.push(item[1]);
        previous = item.index + item[0].length;
      }
      patterns.push({regex: new RegExp(expression + escapeRegex(input.slice(previous)) + '$'), keys, en, zh});
    }
  }
  function translate(text, language, depth = 0) {
    const original = String(text);
    const trimmed = original.trim();
    const exact = direct.get(trimmed);
    let result = exact?.[language];
    // Buttons include decorative arrows; keep the symbols outside the copy.
    if (result === undefined) {
      const decorated = trimmed.match(/^([↺↥↓]\s*)?(.+?)(\s*[↗→])?$/);
      const label = decorated && direct.get(decorated[2]);
      if (label) result = (decorated[1] || '') + label[language] + (decorated[3] || '');
    }
    if (result === undefined && depth < 4) {
      for (const pattern of patterns) {
        const match = trimmed.match(pattern.regex);
        if (!match) continue;
        const values = Object.fromEntries(pattern.keys.map((key, index) => [key, match[index + 1]]));
        result = pattern[language].replace(/\{(\w+)\}/g, (_, key) =>
          ['status', 'duration', 'prerequisites'].includes(key) ? translate(values[key], language, depth + 1) : values[key]);
        break;
      }
    }
    if (result === undefined) return original;
    return original.slice(0, original.indexOf(trimmed)) + result + original.slice(original.indexOf(trimmed) + trimmed.length);
  }
  let language = 'en';
  try { if (localStorage.getItem('skillbridge.language') === 'zh') language = 'zh'; } catch { /* File mode/private storage may disallow persistence. */ }
  const originals = new WeakMap();
  const attributes = new WeakMap();
  const selector = document.getElementById('display-language');
  const ignored = element => !element || Boolean(element.closest('script,style,textarea,[data-i18n-ignore]'));
  function updateText(node) {
    if (ignored(node.parentElement)) return;
    const value = node.nodeValue;
    const old = originals.get(node);
    const source = old && old.last === value ? old.source : translate(value, 'en');
    const next = translate(source, language);
    originals.set(node, {source, last: next});
    if (value !== next) node.nodeValue = next;
  }
  function updateAttributes(element) {
    if (ignored(element)) return;
    const cached = attributes.get(element) || {};
    for (const name of ['placeholder', 'title', 'aria-label']) {
      if (!element.hasAttribute(name)) continue;
      const value = element.getAttribute(name), old = cached[name];
      const source = old && old.last === value ? old.source : translate(value, 'en');
      const next = translate(source, language);
      cached[name] = {source, last: next};
      if (value !== next) element.setAttribute(name, next);
    }
    attributes.set(element, cached);
  }
  function refresh(root = document.documentElement) {
    if (root.nodeType === 3) { updateText(root); return; }
    if (root.nodeType !== 1 || ignored(root)) return;
    updateAttributes(root);
    root.querySelectorAll('[placeholder],[title],[aria-label]').forEach(updateAttributes);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) updateText(walker.currentNode);
  }
  function setLanguage(value) {
    language = value === 'zh' ? 'zh' : 'en';
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    selector.value = language;
    try { localStorage.setItem('skillbridge.language', language); } catch { /* Session-only switching still works. */ }
    refresh();
  }
  selector.addEventListener('change', () => setLanguage(selector.value));
  new MutationObserver(records => {
    const roots = new Set(records.map(record => record.target));
    roots.forEach(refresh);
  }).observe(document.documentElement, {subtree: true, childList: true, characterData: true, attributes: true,
    attributeFilter: ['placeholder', 'title', 'aria-label']});
  window.SkillBridgeI18n = {setLanguage, refresh, translate, get language() { return language; }};
  setLanguage(language);
})();

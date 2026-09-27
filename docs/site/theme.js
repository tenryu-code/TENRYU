/* Color scheme of the documentation site: automatic (follows the operating
   system through prefers-color-scheme; the default), light, or dark.
   Loaded in <head> without defer, so a stored choice applies before the
   first paint. nav.js calls TenryuTheme.mountToggle() to put the switch in
   the top bar. The GUI manual (gui/manual/) reads the same stored choice. */
(function () {
  "use strict";

  var KEY = "tenryu-docs-theme";
  var root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  var buttons = [];

  function readStored() {
    try {
      var value = window.localStorage.getItem(KEY);
      return value === "light" || value === "dark" ? value : null;
    } catch (err) {
      return null;
    }
  }

  function writeStored(mode) {
    try {
      if (mode) window.localStorage.setItem(KEY, mode);
      else window.localStorage.removeItem(KEY);
    } catch (err) {
      /* storage unavailable: the choice lasts until the page is left */
    }
  }

  function apply(mode) {
    if (mode) root.setAttribute("data-theme", mode);
    else root.removeAttribute("data-theme");
  }

  var current = readStored();
  apply(current);

  function systemMode() {
    return media && media.matches ? "dark" : "light";
  }

  // Automatic -> the opposite of the system setting -> the system setting,
  // chosen explicitly -> automatic, so the first press always changes what
  // the reader sees.
  function nextMode(mode) {
    var system = systemMode();
    var opposite = system === "dark" ? "light" : "dark";
    if (!mode) return opposite;
    if (mode === opposite) return system;
    return null;
  }

  var TEXT = {
    ja: {
      name: "配色",
      mode: { auto: "自動", light: "ライト", dark: "ダーク" },
      detail: { auto: "OS の設定に従う", light: "常にライト", dark: "常にダーク" },
      press: function (next) { return "押すと" + next + "に切り替える"; }
    },
    en: {
      name: "Color scheme",
      mode: { auto: "Auto", light: "Light", dark: "Dark" },
      detail: { auto: "follows the system setting", light: "always light", dark: "always dark" },
      press: function (next) { return "Press to switch to " + next.toLowerCase(); }
    }
  };

  var ICONS = {
    auto: '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M8 1.75a6.25 6.25 0 0 1 0 12.5z" fill="currentColor"/></svg>',
    light: '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><circle cx="8" cy="8" r="3" fill="currentColor"/><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    dark: '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path d="M13.5 10.2A6 6 0 0 1 5.8 2.5a6 6 0 1 0 7.7 7.7z" fill="currentColor"/></svg>'
  };

  function render(button, lang) {
    var text = TEXT[lang] || TEXT.en;
    var key = current || "auto";
    var next = nextMode(current) || "auto";
    var nextName = text.mode[next];
    var description = text.name + ": " + text.mode[key] + " (" + text.detail[key] + "). " + text.press(nextName);
    if (lang === "ja") {
      description = text.name + ": " + text.mode[key] + "（" + text.detail[key] + "）。" + text.press(nextName);
    }
    button.innerHTML = ICONS[key] + '<span class="theme-toggle-label">' + text.name + ": " + text.mode[key] + "</span>";
    button.setAttribute("aria-label", description);
    button.title = description;
  }

  function renderAll() {
    buttons.forEach(function (entry) { render(entry.button, entry.lang); });
  }

  function mountToggle(container, lang) {
    if (!container) return;
    var button = document.createElement("button");
    button.type = "button";
    button.className = "theme-toggle";
    button.addEventListener("click", function () {
      current = nextMode(current);
      writeStored(current);
      apply(current);
      renderAll();
    });
    container.appendChild(button);
    buttons.push({ button: button, lang: lang === "ja" ? "ja" : "en" });
    renderAll();
  }

  // A change of the system setting alters what the next press does; a choice
  // made in another tab of the site applies here too.
  if (media) {
    if (media.addEventListener) media.addEventListener("change", renderAll);
    else if (media.addListener) media.addListener(renderAll);
  }
  window.addEventListener("storage", function (event) {
    if (event.key !== KEY) return;
    current = readStored();
    apply(current);
    renderAll();
  });

  window.TenryuTheme = { mountToggle: mountToggle };
})();

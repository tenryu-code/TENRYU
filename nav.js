(function () {
  "use strict";

  const NAV_TREES = {
    en: [
      {
        title: "Home",
        pages: [
          { id: "home", title: "TENRYU", href: "index.html" }
        ]
      },
      {
        title: "Overview",
        pages: [
          { id: "overview", title: "Overview", href: "overview/index.html" },
          { id: "architecture", title: "Architecture & module map", href: "overview/architecture.html" },
          { id: "numerics-foundations", title: "Numerical foundations", href: "overview/numerics-foundations.html" },
          { id: "timestep", title: "Time-step control & CFL", href: "overview/timestep.html" },
          { id: "mpi", title: "MPI decomposition", href: "overview/mpi.html" },
          { id: "verification-philosophy", title: "Verification philosophy", href: "overview/verification-philosophy.html" },
          { id: "nomenclature", title: "Nomenclature & conventions", href: "overview/nomenclature.html" }
        ]
      },
      {
        title: "Using TENRYU",
        pages: [
          { id: "using-tenryu", title: "Using TENRYU", href: "use/index.html" },
          { id: "requirements", title: "System requirements", href: "use/requirements.html" },
          { id: "building", title: "Building", href: "use/building.html" },
          { id: "running", title: "Running simulations", href: "use/running.html" },
          { id: "namelist", title: "Namelist reference", href: "use/namelist.html" },
          { id: "gui", title: "TENRYU Studio (GUI)", href: "use/gui.html" },
          { id: "examples", title: "Examples & presets", href: "use/examples.html" },
          { id: "outputs", title: "Outputs & analysis", href: "use/outputs/index.html" },
          { id: "outputs-snapshots", title: "Snapshots", href: "use/outputs/snapshots.html", parent: "outputs" },
          { id: "outputs-history", title: "History", href: "use/outputs/history.html", parent: "outputs" },
          { id: "outputs-checkpoints", title: "Checkpoints & restart", href: "use/outputs/checkpoints.html", parent: "outputs" }
        ]
      },
      {
        title: "Physics kernels",
        pages: [
          { id: "physics", title: "Physics kernels", href: "physics/index.html" },
          { id: "hydrodynamics", title: "Lagrangian hydrodynamics", href: "physics/hydrodynamics/index.html" },
          { id: "hydro-discretization", title: "Spatial discretization", href: "physics/hydrodynamics/discretization.html", parent: "hydrodynamics" },
          { id: "hydro-time", title: "Time integration", href: "physics/hydrodynamics/time-integration.html", parent: "hydrodynamics" },
          { id: "hydro-av", title: "Artificial viscosity & hourglass control", href: "physics/hydrodynamics/artificial-viscosity.html", parent: "hydrodynamics" },
          { id: "hydro-momentum", title: "RZ momentum schemes", href: "physics/hydrodynamics/momentum-schemes.html", parent: "hydrodynamics" },
          { id: "viscosity", title: "Plasma viscosity", href: "physics/hydrodynamics/plasma-viscosity.html", parent: "hydrodynamics" },
          { id: "mesh", title: "Meshes & geometry", href: "physics/mesh/index.html" },
          { id: "mesh-1d", title: "1D meshes & regions", href: "physics/mesh/1d.html", parent: "mesh" },
          { id: "mesh-2d", title: "2D RZ mesh families", href: "physics/mesh/2d.html", parent: "mesh" },
          { id: "ale", title: "ALE (arbitrary Lagrangian–Eulerian)", href: "physics/ale/index.html" },
          { id: "ale-rezone", title: "Rezoning", href: "physics/ale/rezone.html", parent: "ale" },
          { id: "ale-remap", title: "Conservative remap", href: "physics/ale/remap.html", parent: "ale" },
          { id: "ale-reale-v2", title: "Reconnection-based ALE (reale_v2)", href: "physics/ale/reale-v2.html", parent: "ale" },
          { id: "radiation", title: "Radiation transport", href: "physics/radiation/index.html" },
          { id: "radiation-fld", title: "Multigroup FLD", href: "physics/radiation/fld.html", parent: "radiation" },
          { id: "radiation-sn", title: "S<sub>N</sub> transport", href: "physics/radiation/sn.html", parent: "radiation" },
          { id: "conduction", title: "Heat conduction", href: "physics/conduction/index.html" },
          { id: "conduction-local", title: "Local model & flux limiting", href: "physics/conduction/local.html", parent: "conduction" },
          { id: "conduction-discretization", title: "Spatial discretization", href: "physics/conduction/discretization.html", parent: "conduction" },
          { id: "conduction-time", title: "Time integration", href: "physics/conduction/time-integration.html", parent: "conduction" },
          { id: "conduction-snb", title: "SNB nonlocal model", href: "physics/conduction/snb.html", parent: "conduction" },
          { id: "conduction-ion", title: "Ion heat conduction", href: "physics/conduction/ion.html", parent: "conduction" },
          { id: "laser", title: "Laser drive & absorption", href: "physics/laser/index.html" },
          { id: "laser-beams", title: "Beams & ray launch", href: "physics/laser/beams.html", parent: "laser" },
          { id: "laser-raytrace", title: "Ray tracing", href: "physics/laser/raytrace.html", parent: "laser" },
          { id: "laser-absorption", title: "Absorption & deposition", href: "physics/laser/absorption.html", parent: "laser" },
          { id: "lpi", title: "Laser–plasma interaction (LPI)", href: "physics/lpi/index.html" },
          { id: "cbet", title: "Cross-beam energy transfer (CBET)", href: "physics/lpi/cbet.html", parent: "lpi" },
          { id: "hot-electrons", title: "Hot electrons", href: "physics/lpi/hot-electrons.html", parent: "lpi" },
          { id: "eos-opacity", title: "EOS, opacity & ionization", href: "physics/eos-opacity/index.html" },
          { id: "eos", title: "Equation of state", href: "physics/eos-opacity/eos.html", parent: "eos-opacity" },
          { id: "ionization", title: "Ionization & effective charge", href: "physics/eos-opacity/ionization.html", parent: "eos-opacity" },
          { id: "opacity", title: "Opacity", href: "physics/eos-opacity/opacity.html", parent: "eos-opacity" },
          { id: "burn", title: "Nuclear burn", href: "physics/burn/index.html" },
          { id: "burn-reactions", title: "Reaction rates & depletion", href: "physics/burn/reactions.html", parent: "burn" },
          { id: "burn-products", title: "Charged-product transport", href: "physics/burn/products.html", parent: "burn" },
          { id: "burn-neutrons", title: "Neutron heating", href: "physics/burn/neutrons.html", parent: "burn" }
        ]
      },
      {
        title: "Verification & benchmarks",
        pages: [
          { id: "verification", title: "Verification & benchmarks", href: "verification/index.html" }
        ]
      },
      {
        title: "Developer guide",
        pages: [
          { id: "develop", title: "Developer guide", href: "develop/index.html" }
        ]
      },
      {
        title: "License",
        pages: [
          { id: "license", title: "License", href: "license.html" }
        ]
      }
    ],
    ja: [
      {
        title: "ホーム",
        pages: [
          { id: "home", title: "TENRYU", href: "index.html" }
        ]
      },
      {
        title: "概要",
        pages: [
          { id: "overview", title: "概要", href: "overview/index.html" },
          { id: "architecture", title: "アーキテクチャとモジュール地図", href: "overview/architecture.html" },
          { id: "numerics-foundations", title: "数値基盤", href: "overview/numerics-foundations.html" },
          { id: "timestep", title: "時間刻み制御と CFL", href: "overview/timestep.html" },
          { id: "mpi", title: "MPI 分割", href: "overview/mpi.html" },
          { id: "verification-philosophy", title: "検証の哲学", href: "overview/verification-philosophy.html" },
          { id: "nomenclature", title: "記号・用語集", href: "overview/nomenclature.html" }
        ]
      },
      {
        title: "TENRYU を使う",
        pages: [
          { id: "using-tenryu", title: "TENRYU を使う", href: "use/index.html" },
          { id: "requirements", title: "動作環境と要求スペック", href: "use/requirements.html" },
          { id: "building", title: "ビルド", href: "use/building.html" },
          { id: "running", title: "シミュレーションの実行", href: "use/running.html" },
          { id: "namelist", title: "namelist リファレンス", href: "use/namelist.html" },
          { id: "gui", title: "TENRYU Studio (GUI)", href: "use/gui.html" },
          { id: "examples", title: "例題とプリセット", href: "use/examples.html" },
          { id: "outputs", title: "出力と解析", href: "use/outputs/index.html" },
          { id: "outputs-snapshots", title: "スナップショット", href: "use/outputs/snapshots.html", parent: "outputs" },
          { id: "outputs-history", title: "履歴", href: "use/outputs/history.html", parent: "outputs" },
          { id: "outputs-checkpoints", title: "チェックポイントと再開", href: "use/outputs/checkpoints.html", parent: "outputs" }
        ]
      },
      {
        title: "物理カーネル",
        pages: [
          { id: "physics", title: "物理カーネル", href: "physics/index.html" },
          { id: "hydrodynamics", title: "ラグランジュ流体", href: "physics/hydrodynamics/index.html" },
          { id: "hydro-discretization", title: "空間離散化", href: "physics/hydrodynamics/discretization.html", parent: "hydrodynamics" },
          { id: "hydro-time", title: "時間積分", href: "physics/hydrodynamics/time-integration.html", parent: "hydrodynamics" },
          { id: "hydro-av", title: "人工粘性と砂時計モードの制御", href: "physics/hydrodynamics/artificial-viscosity.html", parent: "hydrodynamics" },
          { id: "hydro-momentum", title: "RZ 運動量スキーム", href: "physics/hydrodynamics/momentum-schemes.html", parent: "hydrodynamics" },
          { id: "viscosity", title: "プラズマ粘性", href: "physics/hydrodynamics/plasma-viscosity.html", parent: "hydrodynamics" },
          { id: "mesh", title: "メッシュと幾何", href: "physics/mesh/index.html" },
          { id: "mesh-1d", title: "1D メッシュと領域", href: "physics/mesh/1d.html", parent: "mesh" },
          { id: "mesh-2d", title: "2D RZ メッシュ族", href: "physics/mesh/2d.html", parent: "mesh" },
          { id: "ale", title: "ALE（任意ラグランジュ・オイラー法）", href: "physics/ale/index.html" },
          { id: "ale-rezone", title: "リゾーン", href: "physics/ale/rezone.html", parent: "ale" },
          { id: "ale-remap", title: "保存的リマップ", href: "physics/ale/remap.html", parent: "ale" },
          { id: "ale-reale-v2", title: "接続を組み替える ALE（reale_v2）", href: "physics/ale/reale-v2.html", parent: "ale" },
          { id: "radiation", title: "輻射輸送", href: "physics/radiation/index.html" },
          { id: "radiation-fld", title: "多群 FLD", href: "physics/radiation/fld.html", parent: "radiation" },
          { id: "radiation-sn", title: "S<sub>N</sub> 輸送", href: "physics/radiation/sn.html", parent: "radiation" },
          { id: "conduction", title: "熱伝導", href: "physics/conduction/index.html" },
          { id: "conduction-local", title: "局所モデルと流束制限", href: "physics/conduction/local.html", parent: "conduction" },
          { id: "conduction-discretization", title: "空間離散化", href: "physics/conduction/discretization.html", parent: "conduction" },
          { id: "conduction-time", title: "時間積分", href: "physics/conduction/time-integration.html", parent: "conduction" },
          { id: "conduction-snb", title: "SNB 非局所モデル", href: "physics/conduction/snb.html", parent: "conduction" },
          { id: "conduction-ion", title: "イオン熱伝導", href: "physics/conduction/ion.html", parent: "conduction" },
          { id: "laser", title: "レーザー駆動と吸収", href: "physics/laser/index.html" },
          { id: "laser-beams", title: "ビームと光線の打ち出し", href: "physics/laser/beams.html", parent: "laser" },
          { id: "laser-raytrace", title: "光線追跡", href: "physics/laser/raytrace.html", parent: "laser" },
          { id: "laser-absorption", title: "吸収と沈着", href: "physics/laser/absorption.html", parent: "laser" },
          { id: "lpi", title: "LPI（レーザープラズマ相互作用）", href: "physics/lpi/index.html" },
          { id: "cbet", title: "CBET (交差ビームエネルギー転送)", href: "physics/lpi/cbet.html", parent: "lpi" },
          { id: "hot-electrons", title: "ホット電子", href: "physics/lpi/hot-electrons.html", parent: "lpi" },
          { id: "eos-opacity", title: "EOS・不透明度・電離", href: "physics/eos-opacity/index.html" },
          { id: "eos", title: "状態方程式", href: "physics/eos-opacity/eos.html", parent: "eos-opacity" },
          { id: "ionization", title: "電離と有効電荷", href: "physics/eos-opacity/ionization.html", parent: "eos-opacity" },
          { id: "opacity", title: "不透明度", href: "physics/eos-opacity/opacity.html", parent: "eos-opacity" },
          { id: "burn", title: "核燃焼", href: "physics/burn/index.html" },
          { id: "burn-reactions", title: "反応率と燃料の減損", href: "physics/burn/reactions.html", parent: "burn" },
          { id: "burn-products", title: "荷電生成物の輸送", href: "physics/burn/products.html", parent: "burn" },
          { id: "burn-neutrons", title: "中性子加熱", href: "physics/burn/neutrons.html", parent: "burn" }
        ]
      },
      {
        title: "検証とベンチマーク",
        pages: [
          { id: "verification", title: "検証とベンチマーク", href: "verification/index.html" }
        ]
      },
      {
        title: "開発者ガイド",
        pages: [
          { id: "develop", title: "開発者ガイド", href: "develop/index.html" }
        ]
      },
      {
        title: "ライセンス",
        pages: [
          { id: "license", title: "ライセンス", href: "license.html" }
        ]
      }
    ]
  };

  const script = document.currentScript;
  const currentPageId = script ? script.dataset.page : "";
  const rootPrefix = script ? (script.dataset.root || "") : "";
  const lang = document.documentElement.dataset.lang === "ja" ? "ja" : "en";
  const NAV_TREE = NAV_TREES[lang];
  const hrefFromRoot = (href) => rootPrefix + href;
  const flatPages = NAV_TREE.flatMap((group) =>
    group.pages.map((page) => ({ ...page, groupTitle: group.title }))
  );
  const currentPage = flatPages.find((page) => page.id === currentPageId);
  const text = lang === "ja"
    ? {
        home: "ホーム",
        guiManual: "GUI マニュアル",
        externalLinks: "外部リンク",
        language: "言語",
        documentationNavigation: "ドキュメントナビゲーション",
        breadcrumb: "パンくずリスト",
        previous: "前へ",
        next: "次へ",
        footer: "TENRYU ドキュメント — 2026-10-14。",
        subtitle: "輻射流体コード"
      }
    : {
        home: "Home",
        guiManual: "GUI manual",
        externalLinks: "External links",
        language: "Language",
        documentationNavigation: "Documentation navigation",
        breadcrumb: "Breadcrumb",
        previous: "Previous",
        next: "Next",
        footer: "TENRYU documentation — 2026-10-14.",
        subtitle: "Radiation Hydrodynamics Code"
      };

  function languageHref(targetLang) {
    const pageHref = currentPage ? currentPage.href : "index.html";
    return `${rootPrefix}../${targetLang}/${pageHref}`;
  }

  function languageLink(targetLang, label) {
    const current = targetLang === lang ? ' aria-current="page"' : "";
    return `<a href="${languageHref(targetLang)}" hreflang="${targetLang}"${current}>${label}</a>`;
  }

  function renderTopbar() {
    const target = document.getElementById("topbar");
    if (!target) return;

    target.innerHTML = `
      <header class="topbar">
        <a class="brand" href="${hrefFromRoot("index.html")}">
          <span class="wordmark">TENRYU</span>
          <span class="subtitle">${text.subtitle}</span>
        </a>
        <nav class="topbar-links" aria-label="${text.externalLinks}">
          <a href="${rootPrefix}../gui/manual/index_${lang}.html">${text.guiManual}</a>
          <a href="https://github.com/tenryu-code/TENRYU" aria-label="TENRYU on GitHub">GitHub</a>
          <span class="language-switcher" aria-label="${text.language}">
            ${languageLink("en", "English")}<span aria-hidden="true">|</span>${languageLink("ja", "日本語")}
          </span>
          <span class="theme-toggle-slot"></span>
        </nav>
      </header>`;
    const themeSlot = target.querySelector(".theme-toggle-slot");
    if (themeSlot && window.TenryuTheme) window.TenryuTheme.mountToggle(themeSlot, lang);
  }

  // Sidebar groups that the reader has opened, keyed by the id of the group's
  // first page (the same in both languages). Every group starts closed, and
  // following a link keeps each group as the reader left it.
  const NAV_OPEN_KEY = "tenryu-docs-nav-open";

  function readOpenGroups() {
    try {
      const value = JSON.parse(window.localStorage.getItem(NAV_OPEN_KEY) || "[]");
      return new Set(Array.isArray(value) ? value.filter((key) => typeof key === "string") : []);
    } catch (err) {
      return new Set();
    }
  }

  function writeOpenGroups(openGroups) {
    try {
      window.localStorage.setItem(NAV_OPEN_KEY, JSON.stringify([...openGroups]));
    } catch (err) {
      /* storage unavailable: the state lasts until the page is left */
    }
  }

  function renderSidebar() {
    const target = document.getElementById("sidebar");
    if (!target) return;

    const openGroups = readOpenGroups();
    const groups = NAV_TREE.map((group, groupIndex) => {
      // A group with a single page is that page: a top-level link, no toggle.
      if (group.pages.length === 1) {
        const page = group.pages[0];
        const active = page.id === currentPageId;
        return `
        <section class="nav-group nav-single">
          <a class="nav-top-link${active ? " active" : ""}" href="${hrefFromRoot(page.href)}"${active ? ' aria-current="page"' : ""}>${group.title}</a>
        </section>`;
      }

      const key = group.pages[0].id;
      const open = openGroups.has(key);
      const groupHasActivePage = group.pages.some((page) => page.id === currentPageId);
      const links = group.pages.map((page) => {
        const active = page.id === currentPageId;
        return `<a class="nav-link${page.parent ? " nav-child" : ""}${active ? " active" : ""}" href="${hrefFromRoot(page.href)}"${active ? ' aria-current="page"' : ""}>${page.title}</a>`;
      }).join("");

      return `
        <section class="nav-group${open ? "" : " collapsed"}${groupHasActivePage ? " active-group" : ""}" data-group="${key}">
          <button class="nav-group-toggle" type="button" aria-expanded="${open}" aria-controls="nav-group-${groupIndex}">
            <span>${group.title}</span><span class="nav-chevron" aria-hidden="true">▾</span>
          </button>
          <div class="nav-pages" id="nav-group-${groupIndex}">${links}</div>
        </section>`;
    }).join("");

    target.innerHTML = `<nav class="sidebar-nav" aria-label="${text.documentationNavigation}">${groups}</nav>`;
    target.querySelectorAll(".nav-group-toggle").forEach((button) => {
      button.addEventListener("click", () => {
        const group = button.closest(".nav-group");
        const collapsed = group.classList.toggle("collapsed");
        button.setAttribute("aria-expanded", String(!collapsed));
        const stored = readOpenGroups();
        if (collapsed) stored.delete(group.dataset.group);
        else stored.add(group.dataset.group);
        writeOpenGroups(stored);
      });
    });
  }

  function persistSidebarScroll() {
    const target = document.getElementById("sidebar");
    if (!target) return;
    const key = "tenryu-sidebar-scroll:" + lang;
    let stored = null;
    try { stored = window.sessionStorage.getItem(key); } catch (err) { return; }
    if (stored !== null) {
      const value = parseInt(stored, 10);
      if (!Number.isNaN(value)) target.scrollTop = value;
    }
    target.addEventListener("scroll", () => {
      try { window.sessionStorage.setItem(key, String(target.scrollTop)); } catch (err) { /* storage unavailable */ }
    }, { passive: true });
  }

  function renderBreadcrumb() {
    const target = document.getElementById("breadcrumb");
    if (!target) return;

    target.setAttribute("aria-label", text.breadcrumb);
    if (!currentPage || currentPage.id === "home") {
      target.innerHTML = `<span aria-current="page">${text.home}</span>`;
      return;
    }

    const parts = [`<a href="${hrefFromRoot("index.html")}">${text.home}</a>`];
    const group = NAV_TREE.find((candidate) => candidate.title === currentPage.groupTitle);
    const landing = group ? group.pages[0] : null;
    if (landing && landing.id !== currentPage.id) {
      parts.push(`<a href="${hrefFromRoot(landing.href)}">${group.title}</a>`);
    }
    if (currentPage.parent) {
      const parent = flatPages.find((page) => page.id === currentPage.parent);
      if (parent) {
        parts.push(`<a href="${hrefFromRoot(parent.href)}">${parent.title}</a>`);
      }
    }
    parts.push(`<span aria-current="page">${currentPage.title}</span>`);
    target.innerHTML = parts.join('<span class="breadcrumb-separator" aria-hidden="true">›</span>');
  }

  function renderFooter() {
    const target = document.getElementById("page-footer");
    if (!target) return;

    const currentIndex = flatPages.findIndex((page) => page.id === currentPageId);
    const previous = currentIndex > 0 ? flatPages[currentIndex - 1] : null;
    const next = currentIndex >= 0 && currentIndex < flatPages.length - 1 ? flatPages[currentIndex + 1] : null;
    const previousLink = previous
      ? `<a class="footer-previous" href="${hrefFromRoot(previous.href)}"><span class="footer-label">${text.previous}</span>← ${previous.title}</a>`
      : "";
    const nextLink = next
      ? `<a class="footer-next" href="${hrefFromRoot(next.href)}"><span class="footer-label">${text.next}</span>${next.title} →</a>`
      : "";

    target.className = "page-footer";
    target.innerHTML = `
      <nav class="footer-nav" aria-label="${text.previous} / ${text.next}">${previousLink}${nextLink}</nav>
      <div>${text.footer}</div>`;
  }

  // A table wider than the text column scrolls inside a wrapper
  // (.table-scroll in style.css) instead of widening the page.
  function wrapTables() {
    document.querySelectorAll("main table").forEach((table) => {
      if (table.parentElement.classList.contains("table-scroll")) return;
      const wrapper = document.createElement("div");
      wrapper.className = "table-scroll";
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });
  }

  // A display equation wider than the text column scrolls inside the column
  // (.is-wide in style.css). Only such equations get the class, after MathJax
  // has typeset the page and again when the window is resized, so every
  // equation that fits keeps overflow visible and none of its ink is clipped.
  function markWideEquations() {
    document.querySelectorAll('main mjx-container[display="true"]').forEach((container) => {
      const math = container.querySelector("mjx-math");
      const wide = !!math && math.getBoundingClientRect().width > container.clientWidth + 0.5;
      container.classList.toggle("is-wide", wide);
    });
  }

  function watchEquationWidths() {
    let pending = false;
    window.addEventListener("resize", () => {
      if (pending) return;
      pending = true;
      window.requestAnimationFrame(() => {
        pending = false;
        markWideEquations();
      });
    });
    const afterTypesetting = () => {
      const mathJax = window.MathJax;
      if (mathJax && mathJax.startup && mathJax.startup.promise) {
        // MathJax reports its own startup failure; nothing to mark then.
        mathJax.startup.promise.then(markWideEquations, () => {});
      }
    };
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", afterTypesetting);
    } else {
      afterTypesetting();
    }
  }

  renderTopbar();
  renderSidebar();
  persistSidebarScroll();
  renderBreadcrumb();
  renderFooter();
  wrapTables();
  watchEquationWidths();
})();

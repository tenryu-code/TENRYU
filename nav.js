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
          { id: "outputs", title: "Outputs & analysis", href: "use/outputs.html" }
        ]
      },
      {
        title: "Physics kernels",
        pages: [
          { id: "physics", title: "Physics kernels", href: "physics/index.html" },
          { id: "hydrodynamics", title: "Lagrangian hydrodynamics", href: "physics/hydrodynamics.html" },
          { id: "mesh", title: "Meshes & geometry", href: "physics/mesh/index.html" },
          { id: "mesh-1d", title: "1D meshes & regions", href: "physics/mesh/1d.html", parent: "mesh" },
          { id: "mesh-2d", title: "2D RZ mesh families", href: "physics/mesh/2d.html", parent: "mesh" },
          { id: "ale-remap", title: "ALE: rezoning & remap", href: "physics/mesh/ale-remap.html", parent: "mesh" },
          { id: "radiation", title: "Radiation transport", href: "physics/radiation/index.html" },
          { id: "radiation-fld", title: "Multigroup FLD", href: "physics/radiation/fld.html", parent: "radiation" },
          { id: "radiation-sn", title: "S<sub>N</sub> transport", href: "physics/radiation/sn.html", parent: "radiation" },
          { id: "conduction", title: "Electron & ion conduction", href: "physics/conduction.html" },
          { id: "laser", title: "Laser drive & absorption", href: "physics/laser.html" },
          { id: "cbet", title: "Cross-beam energy transfer (CBET)", href: "physics/cbet.html" },
          { id: "hot-electrons", title: "Hot electrons", href: "physics/hot-electrons.html" },
          { id: "eos-opacity", title: "EOS, opacity & ionization", href: "physics/eos-opacity.html" },
          { id: "burn", title: "Nuclear burn", href: "physics/burn.html" },
          { id: "viscosity", title: "Plasma viscosity", href: "physics/viscosity.html" },
          { id: "mpi", title: "MPI decomposition", href: "physics/mpi.html" }
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
          { id: "outputs", title: "出力と解析", href: "use/outputs.html" }
        ]
      },
      {
        title: "物理カーネル",
        pages: [
          { id: "physics", title: "物理カーネル", href: "physics/index.html" },
          { id: "hydrodynamics", title: "ラグランジュ流体", href: "physics/hydrodynamics.html" },
          { id: "mesh", title: "メッシュと幾何", href: "physics/mesh/index.html" },
          { id: "mesh-1d", title: "1D メッシュと領域", href: "physics/mesh/1d.html", parent: "mesh" },
          { id: "mesh-2d", title: "2D RZ メッシュ族", href: "physics/mesh/2d.html", parent: "mesh" },
          { id: "ale-remap", title: "ALE: リゾーニングとリマップ", href: "physics/mesh/ale-remap.html", parent: "mesh" },
          { id: "radiation", title: "輻射輸送", href: "physics/radiation/index.html" },
          { id: "radiation-fld", title: "多群 FLD", href: "physics/radiation/fld.html", parent: "radiation" },
          { id: "radiation-sn", title: "S<sub>N</sub> 輸送", href: "physics/radiation/sn.html", parent: "radiation" },
          { id: "conduction", title: "電子・イオン熱伝導", href: "physics/conduction.html" },
          { id: "laser", title: "レーザー駆動と吸収", href: "physics/laser.html" },
          { id: "cbet", title: "CBET (交差ビームエネルギー転送)", href: "physics/cbet.html" },
          { id: "hot-electrons", title: "ホット電子", href: "physics/hot-electrons.html" },
          { id: "eos-opacity", title: "EOS・不透明度・電離", href: "physics/eos-opacity.html" },
          { id: "burn", title: "核燃焼", href: "physics/burn.html" },
          { id: "viscosity", title: "プラズマ粘性", href: "physics/viscosity.html" },
          { id: "mpi", title: "MPI 分割", href: "physics/mpi.html" }
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
        footer: "TENRYU ドキュメント — 2026-08-03。",
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
        footer: "TENRYU documentation — 2026-08-03.",
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
        </nav>
      </header>`;
  }

  function renderSidebar() {
    const target = document.getElementById("sidebar");
    if (!target) return;

    const groups = NAV_TREE.map((group, groupIndex) => {
      const groupHasActivePage = group.pages.some((page) => page.id === currentPageId);
      const links = group.pages.map((page) => {
        const active = page.id === currentPageId;
        return `<a class="nav-link${page.parent ? " nav-child" : ""}${active ? " active" : ""}" href="${hrefFromRoot(page.href)}"${active ? ' aria-current="page"' : ""}>${page.title}</a>`;
      }).join("");

      return `
        <section class="nav-group${groupHasActivePage ? " active-group" : ""}">
          <button class="nav-group-toggle" type="button" aria-expanded="true" aria-controls="nav-group-${groupIndex}">
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

  renderTopbar();
  renderSidebar();
  persistSidebarScroll();
  renderBreadcrumb();
  renderFooter();
})();

import { describe, expect, it } from "vitest";
import {
  parseAssistStatus,
  parseDeckLint,
  parseDigest,
  parseGenerateResult,
  parseJournalTail,
  parsePromoteZoning,
  parseZoningReport,
  type Parsed,
} from "../src/core/assist/parse";

function dataOf<T>(parsed: Parsed<T>): T {
  expect(parsed.ok).toBe(true);
  if (!parsed.ok) throw new Error(parsed.error);
  return parsed.data;
}

describe("parseAssistStatus", () => {
  it("maps and sorts roles from a resolved_summary payload", () => {
    const raw =
      "provider preface\n" +
      JSON.stringify({
        enabled: true,
        disabled_by: null,
        config_source: "/home/user/.tenryu/assistant.toml",
        providers: {
          claude: { model: "opus", command_template: "claude -p {prompt_file}" },
          codex: { model: "gpt-5.6", command_template: "codex exec {prompt_file}" },
        },
        roles: {
          forensics: "claude",
          deck_design: "codex",
          explanation: "dry_run",
        },
        budget: {
          max_interventions_per_run: 3,
          max_tokens_per_decision: 30000,
        },
        warnings: ["review independence is degraded"],
      });

    expect(dataOf(parseAssistStatus(raw))).toEqual({
      enabled: true,
      disabledBy: null,
      configSource: "/home/user/.tenryu/assistant.toml",
      roles: [
        { role: "deck_design", provider: "codex", model: "gpt-5.6" },
        { role: "explanation", provider: "dry_run", model: null },
        { role: "forensics", provider: "claude", model: "opus" },
      ],
      warnings: ["review independence is degraded"],
    });
  });

  it("degrades missing optional sections to nulls and empties", () => {
    expect(
      dataOf(
        parseAssistStatus(
          JSON.stringify({
            enabled: false,
            roles: { deck_design: "dry_run" },
          }),
        ),
      ),
    ).toEqual({
      enabled: false,
      disabledBy: null,
      configSource: null,
      roles: [{ role: "deck_design", provider: "dry_run", model: null }],
      warnings: [],
    });
  });

  it("returns an error and preserves malformed raw input", () => {
    const result = parseAssistStatus("not json");
    expect(result.ok).toBe(false);
    expect(result.raw).toBe("not json");
  });
});

describe("parseDeckLint", () => {
  it("maps a run_deck_lint payload", () => {
    const meshPreview = { nr: 40, r_min: 0, r_max: 0.03, n_r_nodes: 41 };
    const ratioDetail = {
      monotonic_ok: true,
      n_cells: 40,
      max_adjacent_dr_ratio: 1.2,
      at_index: 3,
      pin_jump_edges: [],
      n_exempt_edges: 0,
    };
    const defaultsDiff = {
      differs: [{ path: "mesh.nr", value: 40, baseline: 20 }],
      equal_leaf_count: 12,
      only_in_target: [],
      only_in_baseline: [],
      heuristic_note: "leaves equal to the baseline are only PROBABLY defaults",
    };
    const payload = {
      schema: "tenryu.assist.decklint.v0",
      deck: "/tmp/deck.py",
      tenryu: "/srv/TENRYU/build/tenryu",
      validate: { exit_code: 0, ok: true, stderr_tail: "warning tail" },
      mesh_preview: meshPreview,
      lints: [
        { id: "node-monotonic", severity: "hard", ok: true },
        {
          id: "adjacent-dr-ratio",
          severity: "warn",
          ok: true,
          detail: ratioDetail,
        },
      ],
      intent_lock: [
        {
          path: "mesh.nr",
          expected: 40,
          actual: 40,
          ok: true,
        },
        {
          path: "mesh.missing",
          expected: 1,
          actual: null,
          ok: false,
          reason: "path not found",
        },
      ],
      defaults_diff: defaultsDiff,
    };

    expect(dataOf(parseDeckLint(JSON.stringify(payload)))).toEqual({
      toolError: null,
      validate: { ok: true, exitCode: 0, stderrTail: "warning tail" },
      meshPreview,
      lints: [
        {
          id: "node-monotonic",
          severity: "hard",
          ok: true,
          detail: null,
        },
        {
          id: "adjacent-dr-ratio",
          severity: "warn",
          ok: true,
          detail: ratioDetail,
        },
      ],
      intentLock: [
        {
          path: "mesh.nr",
          ok: true,
          expected: 40,
          actual: 40,
          reason: null,
        },
        {
          path: "mesh.missing",
          ok: false,
          expected: 1,
          actual: null,
          reason: "path not found",
        },
      ],
      defaultsDiff,
    });
  });

  it("degrades missing sections to nulls and empties", () => {
    expect(
      dataOf(
        parseDeckLint(
          JSON.stringify({ schema: "tenryu.assist.decklint.v0" }),
        ),
      ),
    ).toEqual({
      toolError: null,
      validate: { ok: null, exitCode: null, stderrTail: "" },
      meshPreview: null,
      lints: [],
      intentLock: null,
      defaultsDiff: null,
    });
  });

  it("maps the validate execution error shape", () => {
    expect(
      dataOf(
        parseDeckLint(
          JSON.stringify({
            schema: "tenryu.assist.decklint.v0",
            deck: "/tmp/deck.py",
            tenryu: "/srv/TENRYU/build/tenryu",
            error: "validate failed to run: timeout",
          }),
        ),
      ),
    ).toEqual({
      toolError: "validate failed to run: timeout",
      validate: { ok: null, exitCode: null, stderrTail: "" },
      meshPreview: null,
      lints: [],
      intentLock: null,
      defaultsDiff: null,
    });
  });

  it("returns an error and preserves malformed raw input", () => {
    const result = parseDeckLint("not json");
    expect(result.ok).toBe(false);
    expect(result.raw).toBe("not json");
  });
});

describe("parseGenerateResult", () => {
  it("maps an accepted result", () => {
    const lint = { schema: "tenryu.assist.decklint.v0", lints: [] };
    expect(
      dataOf(
        parseGenerateResult(
          JSON.stringify({
            status: "accepted",
            deck_path: "/tmp/work/out_deck.py",
            iterations: 2,
            lint,
          }),
        ),
      ),
    ).toEqual({
      status: "accepted",
      question: null,
      iterations: 2,
      deckPath: "/tmp/work/out_deck.py",
      error: null,
      lint,
    });
  });

  it("maps uncertain and exhausted results", () => {
    expect(
      dataOf(
        parseGenerateResult(
          JSON.stringify({
            status: "uncertain",
            question: "What target radius should be used?",
            iterations: 1,
          }),
        ),
      ),
    ).toEqual({
      status: "uncertain",
      question: "What target radius should be used?",
      iterations: 1,
      deckPath: null,
      error: null,
      lint: null,
    });

    const lastLint = { schema: "tenryu.assist.decklint.v0", lints: [] };
    expect(
      dataOf(
        parseGenerateResult(
          JSON.stringify({
            status: "exhausted",
            iterations: 10,
            last_lint: lastLint,
          }),
        ),
      ),
    ).toEqual({
      status: "exhausted",
      question: null,
      iterations: 10,
      deckPath: null,
      error: null,
      lint: lastLint,
    });
  });

  it("degrades an empty object to an unknown result", () => {
    expect(dataOf(parseGenerateResult("{}"))).toEqual({
      status: "unknown",
      question: null,
      iterations: null,
      deckPath: null,
      error: null,
      lint: null,
    });
  });

  it("returns an error and preserves malformed raw input", () => {
    const result = parseGenerateResult("not json");
    expect(result.ok).toBe(false);
    expect(result.raw).toBe("not json");
  });
});

describe("parseDigest", () => {
  it("maps a digest payload and flattens history.series", () => {
    const run = {
      terminated_normally: true,
      steps: 120,
      t_final_s: 1e-9,
      dt_final_s: 1e-12,
    };
    const frozen = {
      path: "/run/config/deck_frozen.json",
      sha256: "abc",
      dimension: "1D_SPH",
      nr: 40,
    };
    const derived = {
      bang_time_proxy_s: 8e-10,
      laser_deposition_centroid_inside_critical_fraction: 0.75,
      dt_collapse_ratio: 0.1,
    };
    const tracks = {
      dt: { t_s: [0, 1e-9], value: [1e-12, 5e-13] },
    };
    const payload = {
      schema: "tenryu.assist.digest.v0",
      output_dir: "/run/outputs/probe",
      run_info: { termination_reason: "t_end_reached" },
      run,
      frozen_config: frozen,
      history: {
        path: "/run/results/probe_history.h5",
        n_samples: 121,
        series: {
          dt: { min: 5e-13, median: 1e-12, final: 5e-13 },
          "energy/E_total": { first: 10, final: 9.9, drift_rel: -0.01 },
        },
        tracks,
        missing: ["implosion/rho_R"],
      },
      derived,
      notes: ["dt min/median exclude the final sample (t_end step clip)"],
    };

    expect(dataOf(parseDigest(JSON.stringify(payload)))).toEqual({
      run,
      frozen,
      derived,
      historyNSamples: 121,
      historyMissing: ["implosion/rho_R"],
      seriesReductions: [
        {
          name: "dt",
          values: { min: 5e-13, median: 1e-12, final: 5e-13 },
        },
        {
          name: "energy/E_total",
          values: { first: 10, final: 9.9, drift_rel: -0.01 },
        },
      ],
      tracks,
      notes: ["dt min/median exclude the final sample (t_end step clip)"],
    });
  });

  it("degrades missing sections to nulls and empties", () => {
    expect(
      dataOf(
        parseDigest(JSON.stringify({ schema: "tenryu.assist.digest.v0" })),
      ),
    ).toEqual({
      run: null,
      frozen: null,
      derived: null,
      historyNSamples: null,
      historyMissing: [],
      seriesReductions: [],
      tracks: null,
      notes: [],
    });
  });

  it("returns an error and preserves malformed raw input", () => {
    const result = parseDigest("not json");
    expect(result.ok).toBe(false);
    expect(result.raw).toBe("not json");
  });
});

describe("parseZoningReport", () => {
  it("maps a complete zoning-report payload", () => {
    const critical = {
      rho_c_gcc: 0.12,
      L_c_min_cm: 0.001,
      track: [{ t_s: 1e-9, r_c_cm: 0.02 }],
    };
    const ablated = { n_cells: 4, n_total_cells: 40 };
    const lintA = {
      threshold_g_cm2: 0.00012,
      max_areal_mass_g_cm2: 0.0001,
      at_index: 35,
      n_violations: 0,
      violating_indices_head: [],
      ok: true,
      caveats: ["snapshot cadence limits L_c minimum"],
    };
    const pressureTrack = {
      track: [{ t_s: 1e-9, p_peak_overdense: 2e15 }],
      uncalibrated_indicator: true,
    };
    const payload = {
      schema: "tenryu.assist.zoning.v0",
      output_dir: "/run/outputs/probe",
      applicable: true,
      reason: null,
      geometry: "1D_SPH",
      n_snapshots_used: 5,
      kappa: 1,
      ablate_margin: 1.2,
      critical,
      ablated,
      lint_a: lintA,
      pressure_track: pressureTrack,
      notes: ["kappa provisional"],
    };

    expect(dataOf(parseZoningReport(JSON.stringify(payload)))).toEqual({
      applicable: true,
      reason: null,
      geometry: "1D_SPH",
      nSnapshotsUsed: 5,
      critical,
      ablated,
      lintA,
      lintAOk: true,
      pressureTrack,
      notes: ["kappa provisional"],
    });
  });

  it("maps the base degraded report", () => {
    expect(
      dataOf(
        parseZoningReport(
          JSON.stringify({
            schema: "tenryu.assist.zoning.v0",
            output_dir: "/run/outputs/probe",
            applicable: false,
            reason: "no usable snapshots",
            geometry: null,
            n_snapshots_used: 0,
            critical: null,
            ablated: null,
            lint_a: null,
            pressure_track: null,
            notes: ["run_info.json not found"],
          }),
        ),
      ),
    ).toEqual({
      applicable: false,
      reason: "no usable snapshots",
      geometry: null,
      nSnapshotsUsed: 0,
      critical: null,
      ablated: null,
      lintA: null,
      lintAOk: null,
      pressureTrack: null,
      notes: ["run_info.json not found"],
    });
  });

  it("returns an error and preserves malformed raw input", () => {
    const result = parseZoningReport("not json");
    expect(result.ok).toBe(false);
    expect(result.raw).toBe("not json");
  });
});

describe("parsePromoteZoning", () => {
  it("maps a promotion and preserves remaining scalar fields", () => {
    const band = {
      measure_frac_begin: 0.8,
      measure_frac_end: 1,
      cell_measure_max: 2e-6,
    };
    const evidence = {
      first_ablated_cell: 37,
      band_start_cell: 34,
      margin_cells: 3,
      n_ablated: 3,
      threshold_areal_g_cm2: 0.00012,
      r0_cm: 0.02,
      initial_mass_frac_at_band_start: 0.8,
    };
    const payload = {
      schema: "tenryu.assist.zoning_promote.v0",
      output_dir: "/run/outputs/probe",
      applicable: true,
      kappa: 1,
      rho_c_gcc: 0.12,
      L_c_min_cm: 0.001,
      n_snapshots_used: 5,
      band,
      evidence,
      conservatism_note: "re-probe if the ablation front reaches deeper",
      suggested_patch: '{"measure_frac_begin": 0.8}',
    };

    expect(dataOf(parsePromoteZoning(JSON.stringify(payload)))).toEqual({
      band,
      evidence,
      conservatismNote: "re-probe if the ablation front reaches deeper",
      extra: {
        schema: "tenryu.assist.zoning_promote.v0",
        output_dir: "/run/outputs/probe",
        applicable: true,
        kappa: 1,
        rho_c_gcc: 0.12,
        L_c_min_cm: 0.001,
        n_snapshots_used: 5,
        suggested_patch: '{"measure_frac_begin": 0.8}',
      },
    });
  });

  it("degrades an inapplicable promotion to null sections", () => {
    expect(
      dataOf(
        parsePromoteZoning(
          JSON.stringify({
            applicable: false,
            reason: "no ablated cells observed",
          }),
        ),
      ),
    ).toEqual({
      band: null,
      evidence: null,
      conservatismNote: null,
      extra: {
        applicable: false,
        reason: "no ablated cells observed",
      },
    });
  });

  it("returns an error and preserves malformed raw input", () => {
    const result = parsePromoteZoning("not json");
    expect(result.ok).toBe(false);
    expect(result.raw).toBe("not json");
  });
});

describe("parseJournalTail", () => {
  it("skips bad lines and summarizes valid journal records", () => {
    const records = [
      {
        schema: "tenryu.assist.journal.v0",
        ts: "2026-09-02T01:00:00+00:00",
        kind: "deck_iteration",
        payload: { iteration: 0, deck_sha256: "a" },
        record_sha256: "hash-a",
      },
      {
        schema: "tenryu.assist.journal.v0",
        ts: "2026-09-02T01:00:01+00:00",
        kind: "lint_result",
        payload: { iteration: 0, exit_code: 2 },
        record_sha256: "hash-b",
      },
      {
        schema: "tenryu.assist.journal.v0",
        ts: "2026-09-02T01:00:02+00:00",
        kind: "deck_iteration",
        payload: { iteration: 1, deck_sha256: "b" },
        record_sha256: "hash-c",
      },
    ];
    const text = [
      JSON.stringify(records[0]),
      "garbage line",
      JSON.stringify(records[1]),
      JSON.stringify(records[2]),
    ].join("\n");

    expect(parseJournalTail(text)).toEqual({
      iterations: 2,
      lastKind: "deck_iteration",
      lastTs: "2026-09-02T01:00:02+00:00",
      entries: [
        { ts: "2026-09-02T01:00:00+00:00", kind: "deck_iteration" },
        { ts: "2026-09-02T01:00:01+00:00", kind: "lint_result" },
        { ts: "2026-09-02T01:00:02+00:00", kind: "deck_iteration" },
      ],
    });
  });

  it("returns an empty view for entirely malformed input", () => {
    expect(parseJournalTail("not json")).toEqual({
      iterations: 0,
      lastKind: null,
      lastTs: null,
      entries: [],
    });
  });
});

import { afterAll, afterEach, describe, expect, it } from "vitest";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { AppSettings, Backend, ExecResult } from "@tenryu-common/backend/types";
import { loadDeckText, parseImportEnvironment, rankServerCopies, workingDirectoryCandidates, IMPORT_HARNESS_FILES } from "../src/core/deck/loadDeck";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import { shQuote } from "@tenryu-common/core/ssh";
import { t } from "../src/i18n";
import { defaultFormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { __setBackendForTest, useApp } from "../src/store";

const root = path.resolve("..");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(),"studio-load-test-"));
afterAll(()=>fs.rmSync(tmp,{recursive:true,force:true}));
afterEach(()=>{
  __setBackendForTest(null);
  useApp.setState({pendingDeckImport:null,deckImportSettings:{},importSettings:{venue:"local",workingDirectory:"",environmentText:""}});
});
const source = 'from tenryu_namelist import *\nMain(name="external",t_end=2.5e-9)\nRadiation(enabled=False)\nLaser(enabled=False)\n';

function backend() {
  const calls: string[][] = [];
  const b = {
    assistHarnessDir: async()=>path.join(root,"tools/assist"),
    appConfigDir: async()=>tmp,
    writeLocalText: async(p:string,text:string)=>{fs.writeFileSync(p,text);},
    readLocalText: async(p:string)=>fs.readFileSync(p,"utf8"),
    execLocal: async(argv:string[]):Promise<ExecResult>=>{
      calls.push(argv);
      let script = argv[2];
      if (process.env.TENRYU_PYTHON && script.startsWith("python3 ")) script = `"${process.env.TENRYU_PYTHON}" `+script.slice(8);
      const r = spawnSync("bash",["-c",script],{encoding:"utf8",timeout:60000,maxBuffer:16*1024*1024});
      return {code:r.status ?? 1,stdout:r.stdout ?? "",stderr:r.stderr ?? "",timedOut:false};
    },
  } as Backend;
  return {b,calls};
}
describe("shared file/paste loader",()=>{
  it("v1 headers use the existing restore path without Python",async()=>{
    const {b,calls} = backend();
    const f = defaultFormState();
    expect(await loadDeckText(b,generateDeck(f))).toEqual(f);
    expect(calls).toHaveLength(0);
  });
  it("corrupt headers remain errors and never execute their bodies",async()=>{
    const {b,calls} = backend();
    await expect(loadDeckText(b,'# TENRYU-GUI-STATE: broken\nraise ValueError("should not execute")')).rejects.toThrow();
    expect(calls).toHaveLength(0);
  });
  it("paste works without a server profile or mirror",async()=>{
    const {b,calls} = backend();
    const form = await loadDeckText(b,source);
    expect(form.main.name).toBe("external");
    expect(form.deckImport?.rules.length).toBeGreaterThan(0);
    expect(calls.some(a=>a[2].includes("--repo-root"))).toBe(false);
  },60000);
  it("file-open and paste store actions show the import report",async()=>{
    const {b} = backend();
    const deckPath = path.join(tmp,"quoted ' $().py");
    b.openLocalTextFile = async()=>({name:"external.py",path:deckPath,content:source});
    __setBackendForTest(b);
    useApp.setState({profiles:[],currentProfileId:null,assistLocalRepo:"",deckIoStatus:null});
    useApp.getState().loadForm(defaultFormState());
    await useApp.getState().loadNamelist();
    expect(useApp.getState().pendingDeckImport).toBeNull();
    expect(useApp.getState().deckIoStatus?.kind).toBe("loaded");
    expect(useApp.getState().importReportOpen).toBe(true);
    expect(useApp.getState().namelistPath).toBe(deckPath);
    expect(useApp.getState().form.main.name).toBe("external");
    expect(useApp.getState().form.deckImport?.evaluation?.workingDirectory).toBe(fs.realpathSync(tmp));
    expect(useApp.getState().deckImportSettings[deckPath]?.workingDirectory).toBe(fs.realpathSync(tmp));
    expect(await useApp.getState().loadDeckText(source)).toBeNull();
    expect(useApp.getState().form.main.name).toBe("external");
  },60000);
  it("Python errors preserve the current form and expose the traceback",async()=>{
    const {b} = backend();
    __setBackendForTest(b);
    useApp.getState().loadForm(defaultFormState());
    const before = useApp.getState().form;
    expect(await useApp.getState().loadDeckText('raise ValueError("test failure")')).toContain("ValueError: test failure");
    expect(useApp.getState().form).toBe(before);
    expect(useApp.getState().deckImportBusy).toBe(false);
    expect(useApp.getState().deckImportSettings).toEqual({});
  },60000);
  it("defaults the venue to the selected server unless chosen explicitly and states the settings on failure",async()=>{
    const {b} = backend();
    __setBackendForTest(b);
    const profile = {id:"p1",name:"S",transport:"ssh",host:"h",runDir:"~/r",tenryuBin:"/b"} as ServerProfile;
    useApp.setState({profiles:[profile],currentProfileId:"p1",importSettings:{venue:"local",workingDirectory:"",environmentText:""}});
    useApp.getState().setPendingDeckImport({text:"x",filename:null,name:""});
    expect(useApp.getState().importSettings.venue).toBe("server");
    useApp.getState().setImportSettings({venue:"local",venueExplicit:true});
    useApp.getState().setPendingDeckImport({text:"x",filename:null,name:""});
    expect(useApp.getState().importSettings.venue).toBe("local");
    useApp.setState({profiles:[],currentProfileId:null,importSettings:{venue:"local",workingDirectory:"",environmentText:""}});
    useApp.getState().setPendingDeckImport({text:"x",filename:null,name:""});
    expect(useApp.getState().importSettings.venue).toBe("local");
    const detail = await useApp.getState().loadDeckText('raise ValueError("test failure")');
    expect(detail).toContain(t().deck.importLocal);
    expect(detail).toContain("ValueError: test failure");
  },60000);
  it("missing Python and missing mesh planner have actionable errors",async()=>{
    const {b} = backend();
    b.execLocal = async()=>({code:127,stdout:"",stderr:"python3 is required to import a Python deck",timedOut:false});
    await expect(loadDeckText(b,source)).rejects.toThrow("python3 is required");
    const real = backend().b;
    await expect(loadDeckText(real,'from tools.mesh_planner import plan_mesh\n'+source,null,path.join(tmp,"missing-mirror"))).rejects.toThrow("tools/mesh_planner.py");
  },60000);
  it("preserves functions outside the sampling domain and reports unverified paths",async()=>{
    const {b} = backend();
    const form = await loadDeckText(b,source+'import math\nGeometry(rho=lambda r:math.log(r))\n');
    expect(form.deckImport?.unverified).toEqual([expect.objectContaining({path:["Geometry","rho"],reason:expect.stringContaining("Sampling failed:")})]);
    expect(form.deckImport?.rules).toContainEqual(expect.objectContaining({path:["Geometry","rho"],kind:"passthrough"}));
    expect(generateDeck(form)).toContain('math.log(r)');
  },60000);
  it("uses the chosen cwd and environment, records them, and replays the runtime context",async()=>{
    const {b} = backend();
    const directory = path.join(tmp,"run directory");
    fs.mkdirSync(directory);
    fs.writeFileSync(path.join(directory,"case.json"),JSON.stringify({t_end:2e-9,rho:1.5,plot_every:7}));
    const filename = path.join(root,"tests/tools/fixtures/deck_import/environment.py");
    const source = fs.readFileSync(filename,"utf8");
    const environment = parseImportEnvironment("TENRYU_IMPORT_CONFIG=case.json\nTENRYU_IMPORT_NAME=chosen");
    await expect(loadDeckText(b,source,filename)).rejects.toThrow(/Traceback/);
    await expect(loadDeckText(b,source,filename,null,{environment})).rejects.toThrow(/case.json/);
    const form = await loadDeckText(b,source,filename,null,{workingDirectory:directory,environment});
    expect(form.main.name).toBe("chosen");
    expect(form.deckImport?.evaluation).toMatchObject({venue:"local",workingDirectory:fs.realpathSync(directory),environment});
    const generated = generateDeck(form);
    fs.writeFileSync(path.join(directory,"case.json"),JSON.stringify({t_end:2e-9,rho:2.5,plot_every:9}));
    const runtime = await loadDeckText(b,generated.replace(/^# TENRYU-GUI-STATE: .+\n/m,""),filename,null,
      {workingDirectory:directory,environment:{...environment,TENRYU_IMPORT_NAME:"runtime"}});
    expect(runtime.main.name).toBe("chosen"); // Mapped at import time.
    expect(runtime.deckImport!.source).toContain("_studio_import_prepare");
    expect(runtime.deckImport!.rules.some(r=>r.path.join(".")==="Output.plot_every" && r.reason.includes("instead of 9"))).toBe(true);
    expect(process.env.TENRYU_IMPORT_NAME).toBeUndefined();
  },60000);
  it("parses environment values literally and rejects shell syntax",()=>{
    expect(parseImportEnvironment("A=x=y\nB=$(touch /tmp/never)\nEMPTY=\n")).toEqual({A:"x=y",B:"$(touch /tmp/never)"});
    expect(()=>parseImportEnvironment("export A=x")).toThrow();
  });
  it("persists successful file settings at both save sites, restores them on startup, and caps recent paths",async()=>{
    const {b} = backend();
    let saved: AppSettings = {};
    b.saveSettings = async settings=>{saved=structuredClone(settings);};
    b.getSettings = async()=>saved;
    b.listProfiles = async()=>[{id:"p",name:"server",transport:"ssh",host:"h",runDir:"/srv",tenryuBin:"/bin/tenryu"}];
    __setBackendForTest(b);
    const settings = {venue:"local" as const,workingDirectory:tmp,environmentText:"NIFDS_CONFIG=case.json\nUNUSED="};
    useApp.setState({importSettings:settings,deckImportSettings:Object.fromEntries(Array.from({length:50},(_,i)=>[`/deck${i}.py`,settings]))});
    const filename = path.join(tmp,"remember.py");
    expect(await useApp.getState().loadDeckText(source,filename)).toBeNull();
    expect(saved.deckImportSettings?.[filename]).toEqual({...settings,lastUsedAt:expect.any(Number)});
    expect(Object.keys(saved.deckImportSettings!)).toHaveLength(50);
    expect(saved.deckImportSettings?.["/deck0.py"]).toBeUndefined();
    await useApp.getState().setUiLang("ja");
    expect(saved.deckImportSettings?.[filename]).toEqual({...settings,lastUsedAt:expect.any(Number)});
    // Settings backends may reorder JSON object keys on disk.
    saved.deckImportSettings = Object.fromEntries(Object.entries(saved.deckImportSettings!).reverse());
    useApp.setState({deckImportSettings:{},importSettings:{venue:"server",workingDirectory:"",environmentText:""}});
    await useApp.getState().loadInitial();
    expect(Object.keys(useApp.getState().deckImportSettings).at(-1)).toBe(filename);
    useApp.getState().setPendingDeckImport({text:source,filename,name:"remember.py"});
    expect(useApp.getState().importSettings).toEqual({...settings,venueExplicit:true});
    // A failed attempt must leave the last successful settings intact.
    useApp.getState().setImportSettings({workingDirectory:path.join(tmp,"absent")});
    expect(await useApp.getState().loadDeckText(source,filename)).toContain("does not exist");
    expect(saved.deckImportSettings?.[filename]).toEqual({...settings,lastUsedAt:expect.any(Number)});
  },60000);
  it("adds upload-directory guidance before the generic hint for a server failure with an empty directory",async()=>{
    const {b} = backend();
    const profile: ServerProfile = {id:"p",name:"server",transport:"ssh",host:"h",runDir:"/srv",tenryuBin:"/bin/tenryu"};
    b.uploadText = async()=>{};
    b.exec = async(_,argv)=>({code:argv[2].includes("mktemp") || argv[2].startsWith("rm -rf") ? 0 : 2,
      stdout:argv[2].includes("mktemp") ? "/srv/tenryu-studio-import.ABC123" : JSON.stringify({ok:false,error:"Traceback: missing table"}),stderr:"",timedOut:false});
    await expect(loadDeckText(b,source,null,null,{venue:"server",profile})).rejects.toThrow(
      `Traceback: missing table\n${t().deck.importDefaultDirectoryHint}\n${t().deck.importSettingsHint}`);
  });
  it("lists candidate working directories and ranks server copies by shared path suffix",()=>{
    expect(workingDirectoryCandidates(["/a/b/c/deck.py"])).toEqual(["/a/b/c","/a/b","/a"]);
    expect(workingDirectoryCandidates(["/a/b/deck.py","/a/b/c/other.py"],1)).toEqual(["/a/b","/a","/a/b/c"]);
    expect(rankServerCopies("/Users/me/NIF_DS/simulation/deck/liquid.py",
      ["/srv/TENRYU/examples/nifds/liquid.py","/srv/nifds_simulation/deck/liquid.py","/srv/other/liquid.py"]))
      .toEqual(["/srv/nifds_simulation/deck/liquid.py","/srv/TENRYU/examples/nifds/liquid.py","/srv/other/liquid.py"]);
  });
  it("auto-detects the server working directory from same-named copies and remembers the successful one",async()=>{
    const {b,calls} = backend();
    const profile: ServerProfile = {id:"test",name:"server",transport:"ssh",host:"test-host",runDir:"/srv/runs",tenryuBin:"/old/build/tenryu"};
    const directory = "/srv/runs/tenryu-studio-import.ABC123";
    const uploaded = new Map<string,string>();
    const triedDirectories: string[] = [];
    b.uploadText = async(_,f,text)=>{uploaded.set(f,text);};
    b.exec = async(_,argv)=>{
      const script = argv[2];
      if (script.includes("mktemp")) return {code:0,stdout:directory+"\n",stderr:"",timedOut:false};
      if (script.startsWith("rm -rf")) return {code:0,stdout:"",stderr:"",timedOut:false};
      if (script.startsWith("find ")) return {code:0,stdout:"/srv/TENRYU/examples/nifds/liquid.py\n/srv/nifds_simulation/deck/liquid.py\n",stderr:"",timedOut:false};
      const request = JSON.parse(uploaded.get(directory+"/request.json")!);
      if (request.operation === "record") {
        triedDirectories.push(request.workingDirectory);
        if (request.workingDirectory !== "/srv/nifds_simulation") return {code:2,stdout:JSON.stringify({ok:false,error:"Traceback: missing table"}),stderr:"",timedOut:false};
        return {code:0,stdout:JSON.stringify({ok:true,blocks:{Main:{name:"remote",t_end:2e-9}},workingDirectory:"/srv/nifds_simulation"}),stderr:"",timedOut:false};
      }
      return {code:0,stdout:JSON.stringify({ok:true,rules:[],failures:[]}),stderr:"",timedOut:false};
    };
    const form = await loadDeckText(b,source,"/Users/me/NIF_DS/simulation/deck/liquid.py",null,{venue:"server",profile,autoDetectWorkingDirectory:true});
    expect(calls).toHaveLength(0);
    expect(triedDirectories).toEqual(["/srv/nifds_simulation/deck","/srv/nifds_simulation"]);
    expect(form.deckImport?.evaluation?.workingDirectory).toBe("/srv/nifds_simulation");
  },60000);
  it("reports every automatically tried directory when none works",async()=>{
    const {b} = backend();
    const profile: ServerProfile = {id:"test",name:"server",transport:"ssh",host:"test-host",runDir:"/srv/runs",tenryuBin:"/old/build/tenryu"};
    const directory = "/srv/runs/tenryu-studio-import.ABC123";
    b.uploadText = async()=>{};
    b.exec = async(_,argv)=>{
      const script = argv[2];
      if (script.includes("mktemp")) return {code:0,stdout:directory+"\n",stderr:"",timedOut:false};
      if (script.startsWith("rm -rf")) return {code:0,stdout:"",stderr:"",timedOut:false};
      if (script.startsWith("find ")) return {code:0,stdout:"/srv/nifds_simulation/deck/liquid.py\n",stderr:"",timedOut:false};
      return {code:2,stdout:JSON.stringify({ok:false,error:"Traceback: missing table"}),stderr:"",timedOut:false};
    };
    const result = loadDeckText(b,source,"/Users/me/NIF_DS/simulation/deck/liquid.py",null,{venue:"server",profile,autoDetectWorkingDirectory:true});
    await expect(result).rejects.toThrow("Traceback: missing table");
    await expect(result).rejects.toThrow(t().deck.importAutoDetectTried("/srv/nifds_simulation/deck\n/srv/nifds_simulation\n/srv"));
    await expect(result).rejects.not.toThrow(t().deck.importDefaultDirectoryHint);
  },60000);
  it("ships this app's harness and executes all stages on the selected server",async()=>{
    const {b,calls} = backend();
    const profile: ServerProfile = {id:"test",name:"server",transport:"ssh",host:"test-host",runDir:"/srv/runs",tenryuBin:"/old/build/tenryu"};
    const uploaded = new Map<string,string>();
    const remote: string[] = [];
    const directory = "/srv/runs/tenryu-studio-import.ABC123";
    const stages: string[] = [];
    b.uploadText = async(p,f,text)=>{expect(p).toBe(profile);uploaded.set(f,text);};
    b.exec = async(p,argv)=>{
      expect(p).toBe(profile); remote.push(argv[2]);
      if (argv[2].includes("mktemp")) return {code:0,stdout:directory+"\n",stderr:"",timedOut:false};
      if (argv[2].startsWith("rm -rf")) return {code:0,stdout:"",stderr:"",timedOut:false};
      const request = JSON.parse(uploaded.get(directory+"/request.json")!);
      stages.push(request.operation);
      expect(request.workingDirectory).toBe("/srv/simulation");
      expect(request.environment).toEqual({NIFDS_CONFIG:"cases/reference.json"});
      const result = request.operation === "record" ? {ok:true,blocks:{Main:{name:"remote",t_end:2e-9}},workingDirectory:"/srv/simulation"} : {ok:true,rules:[],failures:[]};
      return {code:0,stdout:JSON.stringify(result),stderr:"",timedOut:false};
    };
    const form = await loadDeckText(b,source+"# tools.mesh_planner\n","/local/deck.py","/srv/repo",{venue:"server",profile,workingDirectory:"/srv/simulation",environment:{NIFDS_CONFIG:"cases/reference.json"}});
    expect(calls).toHaveLength(0);
    expect(stages).toEqual(["record","verify","compare"]);
    for (const name of IMPORT_HARNESS_FILES) expect(uploaded.get(`${directory}/tools/assist/${name}`)).toBe(fs.readFileSync(path.join(root,"tools/assist",name),"utf8"));
    expect(uploaded.get(directory+"/deck.py")).toContain("Main(");
    expect(remote.filter(s=>s.includes("import-deck")).every(s=>s.includes(`--repo-root ${shQuote("/srv/repo")}`) && s.includes(directory+"/tools/assist/assist.py"))).toBe(true);
    expect(remote.at(-1)).toBe(`rm -rf ${shQuote(directory)}`);
    expect(form.deckImport?.evaluation?.profile).toEqual({id:"test",name:"server",host:"test-host"});
  },60000);
  it.each([
    ["missing Python",127,"python3 is required on the import venue",false],
    ["missing directory",2,"Import working directory does not exist: /srv/missing",false],
    ["deck exception",2,"Traceback\nValueError: shell_eos_file must identify an existing TMAT table",false],
    ["timeout",1,"",true],
  ] as const)("reports server %s and cleans up after evaluation",async(kind,code,error,timedOut)=>{
    const {b} = backend();
    const profile: ServerProfile = {id:"test",name:"server",transport:"ssh",host:"test-host",runDir:"/srv/runs",tenryuBin:"/old/tenryu"};
    const directory = "/srv/runs/tenryu-studio-import.ABC123";
    const remote: string[] = [];
    b.uploadText = async()=>{};
    b.exec = async(_,argv)=>{
      remote.push(argv[2]);
      if (argv[2].startsWith("rm -rf")) return {code:0,stdout:"",stderr:"",timedOut:false};
      if (argv[2].includes("mktemp") && kind !== "missing Python") return {code:0,stdout:directory,stderr:"",timedOut:false};
      return {code,stdout:kind === "missing Python" ? "" : JSON.stringify({ok:false,error}),stderr:error,timedOut};
    };
    const result = loadDeckText(b,source,null,null,{venue:"server",profile,workingDirectory:"/srv/missing"});
    await expect(result).rejects.toThrow(timedOut ? t().deck.importTimeout : error);
    if (!timedOut) await expect(result).rejects.toThrow(t().deck.importSettingsHint);
    await expect(result).rejects.not.toThrow(t().deck.importDefaultDirectoryHint);
    if (kind !== "missing Python") expect(remote.at(-1)).toBe(`rm -rf ${shQuote(directory)}`);
    else expect(remote).toHaveLength(1);
  });
});

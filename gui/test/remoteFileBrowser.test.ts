import { Children, isValidElement, type ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";
import RemoteFileBrowser from "../src/ui/RemoteFileBrowser";
import { t } from "../src/i18n";

// Exercise rendered controls and handlers with a completed server listing.
const hooks = vi.hoisted(()=>({states:[] as unknown[],list:vi.fn()}));
vi.mock("react",async importOriginal=>{
  const actual = await importOriginal<typeof import("react")>();
  return {...actual,useEffect:()=>{},useState:()=>[hooks.states.shift(),vi.fn()]};
});
vi.mock("../src/store",()=>({useApp:(selector:any)=>selector({listRemoteDir:hooks.list})}));
function elements(node: unknown): ReactElement<any>[] {
  return Children.toArray(node as any).flatMap(child=>isValidElement(child)
    ? [child,...elements((child.props as any).children)] : []);
}
function render(mode: "file"|"directory", ok=true, busy=false) {
  const listing = {ok,path:"/srv/run",entries:[{name:"cases",dir:true},{name:"config.json",dir:false},{name:"table.h5",dir:false}]};
  hooks.states = ["/srv/run",listing,busy,"/srv/run",true];
  const onPick = vi.fn();
  const list = vi.fn(async(path:string)=>({...listing,path}));
  hooks.list = list;
  const tree = RemoteFileBrowser({title:"Browse",initialPath:"/srv/run",mode,onPick,onClose:vi.fn()});
  return {tree,nodes:elements(tree),onPick,list};
}
describe("remote directory selection",()=>{
  it("shows all files but only selects the current folder; directory entries navigate",()=>{
    const {tree,nodes,onPick,list} = render("directory");
    expect(tree.props.className).toContain("z-[60]");
    expect(nodes.some(node=>node.type==="input" && node.props.type==="checkbox")).toBe(false);
    const file = nodes.find(node=>node.props.children?.[0]==="config.json")!;
    expect(file).toBeDefined();
    file.props.onClick();
    expect(onPick).not.toHaveBeenCalled();
    const choose = nodes.find(node=>node.props.children===t().remoteFs.selectFolder)!;
    expect(choose.props.variant).toBe("primary");
    expect(choose.props.disabled).toBe(false);
    choose.props.onClick();
    expect(onPick).toHaveBeenCalledWith("/srv/run");
    nodes.find(node=>node.props.children?.[0]==="cases")!.props.onClick();
    expect(list).toHaveBeenCalledWith("/srv/run/cases");
  });
  it.each([[false,false],[true,true]])("disables selection for ok=%s busy=%s",(ok,busy)=>{
    const {nodes} = render("directory",ok,busy);
    expect(nodes.find(node=>node.props.children===t().remoteFs.selectFolder)!.props.disabled).toBe(true);
  });
  it("keeps the existing h5 file selection mode",()=>{
    const {nodes,onPick} = render("file");
    expect(nodes.some(node=>node.props.children?.[0]==="config.json")).toBe(false);
    nodes.find(node=>node.props.children?.[0]==="table.h5")!.props.onClick();
    expect(onPick).toHaveBeenCalledWith("/srv/run/table.h5");
    expect(nodes.some(node=>node.type==="input" && node.props.type==="checkbox")).toBe(true);
  });
});

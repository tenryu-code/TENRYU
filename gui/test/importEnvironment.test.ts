import { describe, expect, it } from "vitest";
import { appendImportEnvironment, discoverImportEnvironment } from "../src/core/deck/importEnvironment";

describe("import environment discovery",()=>{
  it("finds supported literal reads in order and deduplicates them",()=>{
    expect(discoverImportEnvironment(`
environ.get("BEFORE_IMPORT")
os.environ.get('NIFDS_CONFIG')
os.environ["TABLES"]
os.getenv("MODE", "default")
from os import (path, environ)
environ.get('BARE', None)
environ["TABLES"]
os.getenv("NIFDS_CONFIG")
if os.environ["FLAG"] == "yes": pass
`)).toEqual(["NIFDS_CONFIG","TABLES","MODE","BARE","FLAG"]);
  });
  it("ignores comments, strings, writes and computed names",()=>{
    expect(discoverImportEnvironment(`
# os.getenv("COMMENT")
example = '''os.environ["DOCSTRING"]'''
example = "os.getenv('STRING')"
os.environ["WRITE"] = "value"
os.getenv(variable)
os.getenv("PREFIX" + suffix)
obj.os.getenv("OTHER_OBJECT")
from os import environ as env
environ.get("NOT_IMPORTED")
`)).toEqual([]);
  });
  it("appends only absent names without overwriting values or invalid input",()=>{
    const before = "NIFDS_CONFIG=cases/reference.json\nEMPTY=\nbad input";
    const result = appendImportEnvironment(before,["NIFDS_CONFIG","EMPTY","NEW","NEW"]);
    expect(result).toBe(before+"\nNEW=");
    expect(appendImportEnvironment(result,["NEW"])).toBe(result);
    expect(appendImportEnvironment("",["A","B"])).toBe("A=\nB=");
  });
});

import { describe, it, expect } from "vitest";
import { MusicalKey } from "@domain/value-objects/MusicalKey";

describe("MusicalKey", () => {
  it("should parse standard notation 'Am'", () => {
    const key = MusicalKey.parse("Am");
    expect(key.note).toBe("A");
    expect(key.mode).toBe("minor");
  });

  it("should parse standard notation 'C'", () => {
    const key = MusicalKey.parse("C");
    expect(key.note).toBe("C");
    expect(key.mode).toBe("major");
  });

  it("should parse Camelot notation '8A'", () => {
    const key = MusicalKey.parse("8A");
    expect(key.toCamelot()).toBe("8A");
  });

  it("should parse Camelot notation '11B'", () => {
    const key = MusicalKey.parse("11B");
    expect(key.toCamelot()).toBe("11B");
  });

  it("should convert Am to Camelot 8A", () => {
    const key = MusicalKey.parse("Am");
    expect(key.toCamelot()).toBe("8A");
  });

  it("should convert C to Camelot 8B", () => {
    const key = MusicalKey.parse("C");
    expect(key.toCamelot()).toBe("8B");
  });

  it("should reject invalid key strings", () => {
    expect(() => MusicalKey.parse("Z")).toThrow();
    expect(() => MusicalKey.parse("")).toThrow();
    expect(() => MusicalKey.parse("13A")).toThrow();
  });

  it("should be equal to same key", () => {
    const a = MusicalKey.parse("Am");
    const b = MusicalKey.parse("8A");
    expect(a.equals(b)).toBe(true);
  });
});

import { CamelotPosition } from "./CamelotPosition";

type Mode = "major" | "minor";

interface KeyInfo {
  note: string;
  mode: Mode;
  camelot: string;
}

const KEY_MAP: KeyInfo[] = [
  { note: "Ab", mode: "minor", camelot: "1A" },
  { note: "B",  mode: "major", camelot: "1B" },
  { note: "Eb", mode: "minor", camelot: "2A" },
  { note: "F#", mode: "major", camelot: "2B" },
  { note: "Bb", mode: "minor", camelot: "3A" },
  { note: "Db", mode: "major", camelot: "3B" },
  { note: "F",  mode: "minor", camelot: "4A" },
  { note: "Ab", mode: "major", camelot: "4B" },
  { note: "C",  mode: "minor", camelot: "5A" },
  { note: "Eb", mode: "major", camelot: "5B" },
  { note: "G",  mode: "minor", camelot: "6A" },
  { note: "Bb", mode: "major", camelot: "6B" },
  { note: "D",  mode: "minor", camelot: "7A" },
  { note: "F",  mode: "major", camelot: "7B" },
  { note: "A",  mode: "minor", camelot: "8A" },
  { note: "C",  mode: "major", camelot: "8B" },
  { note: "E",  mode: "minor", camelot: "9A" },
  { note: "G",  mode: "major", camelot: "9B" },
  { note: "B",  mode: "minor", camelot: "10A" },
  { note: "D",  mode: "major", camelot: "10B" },
  { note: "F#", mode: "minor", camelot: "11A" },
  { note: "A",  mode: "major", camelot: "11B" },
  { note: "Db", mode: "minor", camelot: "12A" },
  { note: "E",  mode: "major", camelot: "12B" },
];

export class MusicalKey {
  private constructor(
    public readonly note: string,
    public readonly mode: Mode,
    private readonly camelot: string,
  ) {}

  static parse(input: string): MusicalKey {
    if (!input) throw new Error("Empty key string");

    // Try Camelot notation first (e.g., "8A", "11B")
    const camelotMatch = input.match(/^(\d{1,2})([AB])$/);
    if (camelotMatch) {
      const num = parseInt(camelotMatch[1] ?? "", 10);
      if (num < 1 || num > 12) throw new Error(`Invalid Camelot number: ${num}`);
      const entry = KEY_MAP.find((k) => k.camelot === input);
      if (!entry) throw new Error(`Invalid Camelot notation: ${input}`);
      return new MusicalKey(entry.note, entry.mode, entry.camelot);
    }

    // Try standard notation (e.g., "Am", "C", "F#m")
    const stdMatch = input.match(/^([A-G][b#]?)(m)?$/);
    if (stdMatch) {
      const note = stdMatch[1] ?? "";
      const mode: Mode = stdMatch[2] === "m" ? "minor" : "major";
      const entry = KEY_MAP.find((k) => k.note === note && k.mode === mode);
      if (!entry) throw new Error(`Unknown key: ${input}`);
      return new MusicalKey(entry.note, entry.mode, entry.camelot);
    }

    throw new Error(`Cannot parse key: ${input}`);
  }

  toCamelot(): string {
    return this.camelot;
  }

  toCamelotPosition(): CamelotPosition {
    return CamelotPosition.create(this.camelot);
  }

  equals(other: MusicalKey): boolean {
    return this.camelot === other.camelot;
  }

  toString(): string {
    return `${this.note}${this.mode === "minor" ? "m" : ""}`;
  }
}

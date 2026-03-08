export class AudioFingerprint {
  private constructor(
    public readonly fileHash: string,
    public readonly chromaprint?: string,
  ) {}

  static create(fileHash: string, chromaprint?: string): AudioFingerprint {
    if (!fileHash) throw new Error("File hash is required");
    return new AudioFingerprint(fileHash, chromaprint);
  }

  equals(other: AudioFingerprint): boolean {
    return this.fileHash === other.fileHash;
  }
}

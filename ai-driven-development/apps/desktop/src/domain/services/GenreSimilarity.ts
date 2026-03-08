export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i]!;
    const bi = b[i]!;
    dot += ai * bi;
    magA += ai * ai;
    magB += bi * bi;
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  if (denom === 0) return 0;
  return dot / denom;
}

export function computeGenreScore(
  embeddingA: number[] | undefined,
  embeddingB: number[] | undefined,
): number {
  if (!embeddingA || !embeddingB) return 50;
  const sim = cosineSimilarity(embeddingA, embeddingB);
  // Map cosine similarity [-1, 1] to score [0, 100]
  return Math.round(((sim + 1) / 2) * 100);
}

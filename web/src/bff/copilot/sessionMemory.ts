type SessionTurn = {
  ts: number;
  question: string;
  answer: string;
  intent: string;
};

type SessionRecord = {
  updatedAt: number;
  turns: SessionTurn[];
};

const SESSION_TTL_MS = 30 * 60 * 1000;
const MAX_TURNS = 8;

declare global {
  // eslint-disable-next-line no-var
  var __specCopilotSessions: Map<string, SessionRecord> | undefined;
}

function getStore(): Map<string, SessionRecord> {
  if (!globalThis.__specCopilotSessions) {
    globalThis.__specCopilotSessions = new Map<string, SessionRecord>();
  }
  return globalThis.__specCopilotSessions;
}

function cleanup(store: Map<string, SessionRecord>, now: number) {
  for (const [key, value] of store.entries()) {
    if (now - value.updatedAt > SESSION_TTL_MS) {
      store.delete(key);
    }
  }
}

export function getSessionTurns(sessionKey: string): SessionTurn[] {
  const now = Date.now();
  const store = getStore();
  cleanup(store, now);
  return store.get(sessionKey)?.turns ?? [];
}

export function appendSessionTurn(sessionKey: string, turn: SessionTurn): void {
  const now = Date.now();
  const store = getStore();
  cleanup(store, now);
  const record = store.get(sessionKey) ?? { updatedAt: now, turns: [] };
  const turns = [...record.turns, turn].slice(-MAX_TURNS);
  store.set(sessionKey, { updatedAt: now, turns });
}

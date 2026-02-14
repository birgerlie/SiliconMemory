/**
 * Silicon Memory API client.
 * All 19 REST endpoints wrapped with shared auth headers.
 */

const API_BASE = '/api/v1';

function getHeaders() {
  const store = Alpine.store('auth');
  return {
    'Content-Type': 'application/json',
    'X-User-Id': store.userId,
    'X-Tenant-Id': store.tenantId,
  };
}

async function request(method, path, body) {
  const opts = { method, headers: getHeaders() };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

window.api = {
  // Health
  health:  ()           => request('GET', '/health'),
  status:  ()           => request('GET', '/status'),

  // Memory
  store:   (body)       => request('POST', '/store', body),
  recall:  (body)       => request('POST', '/recall', body),
  query:   (body)       => request('POST', '/query', body),
  getItem: (type, id)   => request('GET', `/memory/${type}/${id}`),

  // Working Memory
  workingGetAll: ()          => request('GET', '/working'),
  workingSet:    (key, body) => request('PUT', `/working/${key}`, body),
  workingDelete: (key)       => request('DELETE', `/working/${key}`),

  // Decisions
  decisionStore:  (body) => request('POST', '/decisions', body),
  decisionSearch: (body) => request('POST', '/decisions/search', body),

  // Ingestion
  ingest: (body) => request('POST', '/ingest', body),

  // Reflection
  reflect: (body) => request('POST', '/reflect', body),

  // Entities
  entitiesResolve:   (body) => request('POST', '/entities/resolve', body),
  entitiesRegister:  (body) => request('POST', '/entities/register', body),
  entitiesBootstrap: (body) => request('POST', '/entities/bootstrap', body),
  entitiesLearn:     ()     => request('POST', '/entities/learn'),
  entitiesRules:     ()     => request('GET', '/entities/rules'),

  // Security
  forget: (body) => request('POST', '/forget', body),
};

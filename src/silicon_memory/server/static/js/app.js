/**
 * Silicon Memory — Alpine.js stores and initialization.
 */

document.addEventListener('alpine:init', () => {

  // --- Auth store (persisted to localStorage) ---
  Alpine.store('auth', {
    userId:   localStorage.getItem('sm_user_id')   || 'default',
    tenantId: localStorage.getItem('sm_tenant_id') || 'default',
    save() {
      localStorage.setItem('sm_user_id', this.userId);
      localStorage.setItem('sm_tenant_id', this.tenantId);
    },
  });

  // --- Toast store ---
  Alpine.store('toast', {
    items: [],
    show(msg, type = 'info') {
      const id = Date.now();
      this.items.push({ id, msg, type });
      setTimeout(() => {
        this.items = this.items.filter(t => t.id !== id);
      }, 4000);
    },
    success(msg) { this.show(msg, 'success'); },
    error(msg)   { this.show(msg, 'error'); },
    info(msg)    { this.show(msg, 'info'); },
  });

  // --- Router store ---
  Alpine.store('router', {
    page: 'dashboard',
    init() {
      this.page = (location.hash.slice(1) || 'dashboard');
      window.addEventListener('hashchange', () => {
        this.page = location.hash.slice(1) || 'dashboard';
      });
    },
    go(page) {
      location.hash = page;
    },
  });
});

// Navigation items
window.navItems = [
  { page: 'dashboard',      label: 'Dashboard',      icon: 'chart-bar' },
  { page: 'store',          label: 'Store',           icon: 'plus-circle' },
  { page: 'recall',         label: 'Recall',          icon: 'search' },
  { page: 'query',          label: 'Query',           icon: 'filter' },
  { page: 'detail',         label: 'Detail',          icon: 'document-text' },
  { page: 'working-memory', label: 'Working Memory',  icon: 'clock' },
  { page: 'decisions',      label: 'Decisions',       icon: 'light-bulb' },
  { page: 'ingestion',      label: 'Ingestion',       icon: 'inbox-in' },
  { page: 'entities',       label: 'Entities',        icon: 'user-group' },
  { page: 'reflection',     label: 'Reflection',      icon: 'sparkles' },
  { page: 'security',       label: 'Security',        icon: 'shield-check' },
];

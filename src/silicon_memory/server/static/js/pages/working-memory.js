/**
 * Working Memory page — CRUD table for key-value working memory.
 */
document.addEventListener('alpine:init', () => {
  Alpine.data('workingMemoryPage', () => ({
    entries: {},
    loading: false,
    newKey: '',
    newValue: '',
    newTTL: 300,

    async init() {
      await this.refresh();
    },

    async refresh() {
      this.loading = true;
      try {
        this.entries = await api.workingGetAll();
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.loading = false;
      }
    },

    entryList() {
      return Object.entries(this.entries).map(([k, v]) => ({ key: k, value: v }));
    },

    async add() {
      if (!this.newKey.trim()) return;
      let val = this.newValue;
      try { val = JSON.parse(val); } catch {}
      try {
        await api.workingSet(this.newKey, {
          value: val,
          ttl_seconds: parseInt(this.newTTL) || 300,
        });
        Alpine.store('toast').success(`Set "${this.newKey}"`);
        this.newKey = '';
        this.newValue = '';
        this.newTTL = 300;
        await this.refresh();
      } catch (e) {
        Alpine.store('toast').error(e.message);
      }
    },

    async remove(key) {
      try {
        await api.workingDelete(key);
        Alpine.store('toast').success(`Deleted "${key}"`);
        await this.refresh();
      } catch (e) {
        Alpine.store('toast').error(e.message);
      }
    },

    formatValue(v) {
      if (typeof v === 'object') return JSON.stringify(v, null, 2);
      return String(v);
    },
  }));
});

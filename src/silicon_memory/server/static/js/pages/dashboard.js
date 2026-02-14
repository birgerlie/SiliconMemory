/**
 * Dashboard page — health + status polling.
 */
document.addEventListener('alpine:init', () => {
  Alpine.data('dashboardPage', () => ({
    health: null,
    status: null,
    loading: true,
    error: null,
    interval: null,

    async init() {
      await this.refresh();
      this.interval = setInterval(() => this.refresh(), 30000);
    },

    destroy() {
      if (this.interval) clearInterval(this.interval);
    },

    async refresh() {
      this.loading = true;
      this.error = null;
      try {
        const [h, s] = await Promise.all([api.health(), api.status()]);
        this.health = h;
        this.status = s;
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    formatUptime(seconds) {
      if (!seconds) return '—';
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = Math.floor(seconds % 60);
      return h > 0 ? `${h}h ${m}m ${s}s` : m > 0 ? `${m}m ${s}s` : `${s}s`;
    },
  }));
});

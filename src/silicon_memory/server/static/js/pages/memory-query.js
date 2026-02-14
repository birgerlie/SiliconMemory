/**
 * Query page — semantic search over beliefs.
 */
document.addEventListener('alpine:init', () => {
  Alpine.data('queryPage', () => ({
    query: '',
    limit: 10,
    min_confidence: 0.0,
    loading: false,
    result: null,

    async submit() {
      if (!this.query.trim()) return;
      this.loading = true;
      this.result = null;
      try {
        this.result = await api.query({
          query: this.query,
          limit: parseInt(this.limit),
          min_confidence: parseFloat(this.min_confidence),
        });
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.loading = false;
      }
    },

    statusBadge(status) {
      const map = {
        provisional: 'badge-provisional',
        validated: 'badge-validated',
        contested: 'badge-contested',
        rejected: 'badge-rejected',
      };
      return map[status] || 'badge-provisional';
    },
  }));
});

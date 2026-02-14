/**
 * Recall page — recall across all memory types with tabbed results.
 */
document.addEventListener('alpine:init', () => {
  Alpine.data('recallPage', () => ({
    query: '',
    showAdvanced: false,
    opts: {
      max_facts: 20,
      max_experiences: 10,
      max_procedures: 5,
      min_confidence: 0.3,
      salience_profile: '',
    },
    loading: false,
    result: null,
    activeTab: 'facts',

    async submit() {
      if (!this.query.trim()) return;
      this.loading = true;
      this.result = null;
      try {
        const body = { query: this.query, ...this.opts };
        if (!body.salience_profile) delete body.salience_profile;
        this.result = await api.recall(body);
        this.activeTab = 'facts';
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.loading = false;
      }
    },

    tabCount(tab) {
      if (!this.result) return 0;
      if (tab === 'facts') return this.result.facts?.length || 0;
      if (tab === 'experiences') return this.result.experiences?.length || 0;
      if (tab === 'procedures') return this.result.procedures?.length || 0;
      if (tab === 'working') return Object.keys(this.result.working_context || {}).length;
      return 0;
    },

    confidenceColor(c) {
      if (c >= 0.8) return 'bg-green-500';
      if (c >= 0.5) return 'bg-yellow-500';
      return 'bg-red-400';
    },
  }));
});

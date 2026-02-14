/**
 * Decisions page — record and search decisions.
 */
document.addEventListener('alpine:init', () => {
  Alpine.data('decisionsPage', () => ({
    activeTab: 'record',
    // Record
    form: { title: '', description: '', tags: '' },
    storeLoading: false,
    storeResult: null,
    // Search
    searchQuery: '',
    searchLimit: 10,
    searchLoading: false,
    searchResults: null,

    async submitRecord() {
      if (!this.form.title.trim()) return;
      this.storeLoading = true;
      this.storeResult = null;
      try {
        this.storeResult = await api.decisionStore({
          title: this.form.title,
          description: this.form.description,
          tags: this.form.tags ? this.form.tags.split(',').map(t => t.trim()).filter(Boolean) : [],
        });
        Alpine.store('toast').success('Decision recorded');
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.storeLoading = false;
      }
    },

    async submitSearch() {
      if (!this.searchQuery.trim()) return;
      this.searchLoading = true;
      this.searchResults = null;
      try {
        this.searchResults = await api.decisionSearch({
          query: this.searchQuery,
          limit: parseInt(this.searchLimit),
        });
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.searchLoading = false;
      }
    },
  }));
});

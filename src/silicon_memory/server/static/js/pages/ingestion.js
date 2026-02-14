/**
 * Ingestion page — ingest content from various sources.
 */
document.addEventListener('alpine:init', () => {
  Alpine.data('ingestionPage', () => ({
    form: {
      source_type: 'document',
      content: '',
    },
    loading: false,
    result: null,

    async submit() {
      if (!this.form.content.trim()) return;
      this.loading = true;
      this.result = null;
      try {
        this.result = await api.ingest({
          source_type: this.form.source_type,
          content: this.form.content,
        });
        Alpine.store('toast').success('Content ingested');
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.loading = false;
      }
    },

    reset() {
      this.form.content = '';
      this.result = null;
    },
  }));
});

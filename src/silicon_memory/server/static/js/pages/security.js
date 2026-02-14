/**
 * Security page — GDPR forget / data deletion.
 */
document.addEventListener('alpine:init', () => {
  Alpine.data('securityPage', () => ({
    form: {
      scope: 'entity',
      entity_id: '',
      session_id: '',
      topics: '',
      query: '',
      reason: '',
    },
    loading: false,
    result: null,
    confirmAll: false,

    async submit() {
      if (this.form.scope === 'all' && !this.confirmAll) {
        Alpine.store('toast').error('You must confirm deletion of all data');
        return;
      }
      this.loading = true;
      this.result = null;
      try {
        const body = {
          scope: this.form.scope,
          reason: this.form.reason || undefined,
        };
        if (this.form.scope === 'entity') body.entity_id = this.form.entity_id;
        if (this.form.scope === 'session') body.session_id = this.form.session_id;
        if (this.form.scope === 'topic') {
          body.topics = this.form.topics.split(',').map(t => t.trim()).filter(Boolean);
        }
        if (this.form.scope === 'query') body.query = this.form.query;
        this.result = await api.forget(body);
        Alpine.store('toast').success(`Deleted ${this.result.deleted_count} items`);
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.loading = false;
      }
    },
  }));
});

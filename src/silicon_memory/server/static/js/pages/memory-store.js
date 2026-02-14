/**
 * Store page — store beliefs, experiences, procedures.
 */
document.addEventListener('alpine:init', () => {
  Alpine.data('storePage', () => ({
    form: {
      type: 'auto',
      content: '',
      confidence: 0.5,
      tags: '',
      subject: '',
      predicate: '',
      object: '',
      outcome: '',
      session_id: '',
      name: '',
      description: '',
      trigger: '',
      steps: '',
    },
    loading: false,
    result: null,

    async submit() {
      if (!this.form.content.trim()) {
        Alpine.store('toast').error('Content is required');
        return;
      }
      this.loading = true;
      this.result = null;
      try {
        const body = {
          type: this.form.type,
          content: this.form.content,
          confidence: parseFloat(this.form.confidence),
          tags: this.form.tags ? this.form.tags.split(',').map(t => t.trim()).filter(Boolean) : [],
        };
        if (this.form.type === 'belief' || this.form.type === 'auto') {
          if (this.form.subject) body.subject = this.form.subject;
          if (this.form.predicate) body.predicate = this.form.predicate;
          if (this.form.object) body.object = this.form.object;
        }
        if (this.form.type === 'experience') {
          if (this.form.outcome) body.outcome = this.form.outcome;
          if (this.form.session_id) body.session_id = this.form.session_id;
        }
        if (this.form.type === 'procedure') {
          if (this.form.name) body.name = this.form.name;
          if (this.form.description) body.description = this.form.description;
          if (this.form.trigger) body.trigger = this.form.trigger;
          if (this.form.steps) body.steps = this.form.steps.split('\n').filter(Boolean);
        }
        this.result = await api.store(body);
        Alpine.store('toast').success(`Stored as ${this.result.type} (${this.result.id.slice(0, 8)}…)`);
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.loading = false;
      }
    },

    reset() {
      this.form = {
        type: 'auto', content: '', confidence: 0.5, tags: '',
        subject: '', predicate: '', object: '', outcome: '', session_id: '',
        name: '', description: '', trigger: '', steps: '',
      };
      this.result = null;
    },
  }));
});

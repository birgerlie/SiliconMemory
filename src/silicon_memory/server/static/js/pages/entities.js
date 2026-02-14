/**
 * Entities page — resolve, register, bootstrap, rules/learn.
 */
document.addEventListener('alpine:init', () => {
  Alpine.data('entitiesPage', () => ({
    activeTab: 'resolve',

    // Resolve
    resolveText: '',
    resolveLoading: false,
    resolveResult: null,

    // Register
    regForm: { alias: '', canonical_id: '', entity_type: 'person' },
    regLoading: false,

    // Bootstrap
    bootstrapText: '',
    bootstrapLoading: false,
    bootstrapResult: null,

    // Rules
    rules: null,
    rulesLoading: false,
    learnLoading: false,

    async submitResolve() {
      if (!this.resolveText.trim()) return;
      this.resolveLoading = true;
      this.resolveResult = null;
      try {
        this.resolveResult = await api.entitiesResolve({ text: this.resolveText });
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.resolveLoading = false;
      }
    },

    async submitRegister() {
      if (!this.regForm.alias.trim() || !this.regForm.canonical_id.trim()) return;
      this.regLoading = true;
      try {
        await api.entitiesRegister(this.regForm);
        Alpine.store('toast').success(`Registered "${this.regForm.alias}"`);
        this.regForm = { alias: '', canonical_id: '', entity_type: 'person' };
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.regLoading = false;
      }
    },

    async submitBootstrap() {
      if (!this.bootstrapText.trim()) return;
      this.bootstrapLoading = true;
      this.bootstrapResult = null;
      try {
        this.bootstrapResult = await api.entitiesBootstrap({ text: this.bootstrapText });
        Alpine.store('toast').success('Bootstrap complete');
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.bootstrapLoading = false;
      }
    },

    async loadRules() {
      this.rulesLoading = true;
      try {
        this.rules = await api.entitiesRules();
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.rulesLoading = false;
      }
    },

    async learnRules() {
      this.learnLoading = true;
      try {
        const r = await api.entitiesLearn();
        Alpine.store('toast').success(`Created ${r.rules_created} rules`);
        await this.loadRules();
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.learnLoading = false;
      }
    },
  }));
});

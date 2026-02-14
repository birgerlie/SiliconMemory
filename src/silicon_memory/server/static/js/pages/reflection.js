/**
 * Reflection page — trigger experience → belief reflection.
 */
document.addEventListener('alpine:init', () => {
  Alpine.data('reflectionPage', () => ({
    maxExperiences: 100,
    autoCommit: true,
    loading: false,
    result: null,

    async submit() {
      this.loading = true;
      this.result = null;
      try {
        this.result = await api.reflect({
          max_experiences: parseInt(this.maxExperiences),
          auto_commit: this.autoCommit,
        });
        Alpine.store('toast').success('Reflection complete');
      } catch (e) {
        Alpine.store('toast').error(e.message);
      } finally {
        this.loading = false;
      }
    },
  }));
});

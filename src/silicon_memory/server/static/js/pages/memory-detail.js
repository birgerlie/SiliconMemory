/**
 * Detail page — view a single memory item by type + ID.
 */
document.addEventListener('alpine:init', () => {
  Alpine.data('detailPage', () => ({
    type: 'belief',
    id: '',
    loading: false,
    item: null,
    error: null,

    async submit() {
      if (!this.id.trim()) return;
      this.loading = true;
      this.item = null;
      this.error = null;
      try {
        this.item = await api.getItem(this.type, this.id.trim());
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },
  }));
});

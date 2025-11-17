<script lang="ts">
  import Digit from './Digit.svelte';
  
  export let field: string;
  export let value: string;
</script>

<div class="pair" data-field={field}>
  <Digit index={0} value={value[0]} />
  <Digit index={1} value={value[1]} />
</div>


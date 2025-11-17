<script lang="ts">
  import Hand from './Hand.svelte';
  import Dot from './Dot.svelte';
  
  export let index: number;
  export let rotation: number[];
</script>

<div class="cell" data-index={index}>
  <Hand rotation={rotation[0]} />
  <Hand rotation={rotation[1]} />
  <Dot />
</div>



<script lang="ts">
  import Cell from './Cell.svelte';
  import { cell } from '$lib/utilities/digit';
  
  export let index: number;
  export let value: string;
</script>

<div class="digit" data-index={index}>
  {#each Array(24) as _, i}
    <Cell index={i} rotation={cell(value, i)} />
  {/each}
</div>

<style>
  .digit {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    grid-template-rows: repeat(6, 1fr);
    gap: 2px;
  }
</style>
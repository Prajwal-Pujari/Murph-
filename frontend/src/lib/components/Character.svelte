<script lang="ts">
	import Hand from './Hand.svelte';
	import { cell } from '$lib/utilities/digit';

	export let value: string = ' ';

	const indices = Array.from({ length: 24 }, (_, i) => i);
</script>

<div class="character">
	{#each indices as index}
		{@const [a, b] = cell(value, index)}
		<div class="cell">
			<Hand rotation={a} />
			<Hand rotation={b} />
		</div>
	{/each}
</div>

<style>
	.character {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		grid-template-rows: repeat(6, 1fr);
		gap: 1px;
		width: 28px;
		height: 42px;
	}

	.cell {
		position: relative;
		width: 100%;
		height: 100%;
	}
</style>
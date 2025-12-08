<script lang="ts">
	export let rotation: number;
</script>

<div class="hand" style="rotate: {rotation}deg;"></div>

<style>
	.hand {
		position: absolute;
		top: 50%;
		left: 50%;
		width: 50%;
		height: 2px;
		background-color: var(--hand-color, oklch(94.75% 0.1916 119.15));
		transform-origin: left center;
		translate: 0 -50%;
		border-radius: 1px;
		transition: rotate 0.3s ease-out;
	}
</style>
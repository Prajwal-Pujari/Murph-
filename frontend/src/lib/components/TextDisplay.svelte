<script lang="ts">
	import Character from './Character.svelte';

	export let text: string = '';
	export let visibleChars: number = 0;

	$: words = text.split(' ');
	
	function getCharIndex(wordIndex: number, charIndex: number): number {
		let total = 0;
		for (let i = 0; i < wordIndex; i++) {
			total += words[i].length + 1; // +1 for space
		}
		return total + charIndex;
	}
</script>

<div class="text-display">
	{#each words as word, wordIndex}
		{@const wordStartIndex = getCharIndex(wordIndex, 0)}
		{@const wordEndIndex = wordStartIndex + word.length}
		
		<!-- Only show word if at least one character is visible -->
		{#if visibleChars > wordStartIndex}
			<div class="word">
				{#each word.split('') as char, charIndex}
					{@const globalIndex = getCharIndex(wordIndex, charIndex)}
					
					<!-- Only render character if it should be visible -->
					{#if globalIndex < visibleChars}
						<div class="char-wrapper">
							<Character value={char} />
						</div>
					{/if}
				{/each}
			</div>
			
			<!-- Add space after word if there are more words and next word has started appearing -->
			{#if wordIndex < words.length - 1 && visibleChars > wordEndIndex}
				<div class="space"></div>
			{/if}
		{/if}
	{/each}
</div>

<style>
	.text-display {
		display: flex;
		flex-wrap: wrap;
		justify-content: flex-start;
		align-items: flex-start;
		gap: 6px 4px;
		max-width: 600px;
		max-height: 300px;
		overflow: hidden;
	}

	.word {
		display: flex;
		gap: 2px;
	}

	.space {
		width: 14px;
	}

	.char-wrapper {
		animation: char-appear 0.15s ease-out;
	}

	@keyframes char-appear {
		from {
			opacity: 0;
			transform: scale(0.8);
		}
		to {
			opacity: 1;
			transform: scale(1);
		}
	}
</style>
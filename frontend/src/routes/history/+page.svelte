<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';

	type HistoryPart = {
		type: string;
		content: string;
	};

	type HistoryEntry = {
		id: string;
		parts: HistoryPart[];
	};

	let historyLog: HistoryEntry[] = [];
	let isLoading = true;
	let errorMessage = '';

	function deleteEntry(entryId: string) {
		historyLog = historyLog.filter(entry => entry.id !== entryId);
	}

	onMount(() => {
		const fetchHistory = async () => {
			try {
				const response = await fetch('http://localhost:8000/chat-history');

				if (!response.ok) {
					throw new Error(`Failed to fetch history: ${response.statusText}`);
				}

				historyLog = await response.json();
			} catch (err) {
				if (err instanceof Error) {
					errorMessage = err.message;
				} else {
					errorMessage = 'An unknown error occurred.';
				}
				console.error(err);
			} finally {
				isLoading = false;
			}
		};

		fetchHistory();

		const handleKeyDown = (event: KeyboardEvent) => {
			if (event.ctrlKey && event.shiftKey && event.code === 'KeyQ') {
				event.preventDefault();
				goto('/');
			}
		};

		window.addEventListener('keydown', handleKeyDown);

		return () => {
			window.removeEventListener('keydown', handleKeyDown);
		};
	});

	function getPartClass(type: string): string {
		switch (type.toLowerCase()) {
			case 'user':
				return 'message-user';
			case 'ai':
				return 'message-murph';
			case 'action':
			case 'result':
				return 'message-system';
			default:
				return 'message-unknown';
		}
	}

	function formatType(type: string): string {
		return type.toLowerCase() === 'ai' ? 'MURPH' : type.toUpperCase();
	}
</script>

<svelte:head>
	<title>Chat History</title>
</svelte:head>

<main>
	<div class="header">
		<h1>Chat History</h1>
		<a href="/" class="nav-link">
			<span class="arrow">←</span> Back to Voice Chat
		</a>
	</div>

	<div class="shortcut-hint">
		<kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Q</kbd> to return home
	</div>

	{#if isLoading}
		<div class="status-container">
			<div class="spinner"></div>
			<p class="status">Loading history...</p>
		</div>
	{:else if errorMessage}
		<div class="status-container">
			<p class="status error">{errorMessage}</p>
		</div>
	{:else if historyLog.length === 0}
		<div class="status-container">
			<p class="status empty">No history found. Start talking to MURPH!</p>
		</div>
	{:else}
		<div class="history-container">
			{#each historyLog as entry (entry.id)}
				<div class="chat-entry">
					<button class="delete-btn" on:click={() => deleteEntry(entry.id)} title="Delete entry">
						×
					</button>
					{#each entry.parts as part}
						<div class="message-part {getPartClass(part.type)}">
							<span class="message-label">{formatType(part.type)}</span>
							<div class="message-content">{part.content}</div>
						</div>
					{/each}
				</div>
			{/each}
		</div>
	{/if}
</main>

<style>
	:global(html) {
		overflow-y: scroll;
		scrollbar-width: thin;
		scrollbar-color: rgba(255, 255, 255, 0.1) transparent;
	}

	:global(html::-webkit-scrollbar) {
		width: 6px;
	}

	:global(html::-webkit-scrollbar-track) {
		background: transparent;
	}

	:global(html::-webkit-scrollbar-thumb) {
		background: rgba(255, 255, 255, 0.1);
		border-radius: 3px;
	}

	:global(html::-webkit-scrollbar-thumb:hover) {
		background: rgba(255, 255, 255, 0.15);
	}

	main {
		max-width: 700px;
		margin: 0 auto;
		padding: 60px 30px 40px;
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu,
			Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
		color: #f0f0f0;
		min-height: 100vh;
	}

	.header {
		margin-bottom: 40px;
		text-align: center;
	}

	h1 {
		color: #ffffff;
		font-size: 2.5rem;
		font-weight: 300;
		letter-spacing: 0.5px;
		margin: 0 0 20px 0;
		text-transform: uppercase;
	}

	.nav-link {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		color: #3498db;
		text-decoration: none;
		font-size: 0.95rem;
		font-weight: 400;
		transition: all 0.2s ease;
		padding: 8px 16px;
		border-radius: 6px;
	}

	.nav-link:hover {
		background-color: rgba(52, 152, 219, 0.1);
		transform: translateX(-2px);
	}

	.arrow {
		font-size: 1.2rem;
		transition: transform 0.2s ease;
	}

	.nav-link:hover .arrow {
		transform: translateX(-3px);
	}

	.shortcut-hint {
		text-align: center;
		font-size: 0.85rem;
		color: rgba(255, 255, 255, 0.4);
		margin-bottom: 40px;
		font-weight: 300;
	}

	.shortcut-hint kbd {
		padding: 4px 10px;
		font-size: 0.8em;
		background: linear-gradient(180deg, #3a3a3a 0%, #2a2a2a 100%);
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-radius: 4px;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
		color: rgba(255, 255, 255, 0.8);
		font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
		margin: 0 2px;
		display: inline-block;
	}

	.status-container {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 80px 20px;
		gap: 20px;
	}

	.status {
		font-size: 1.1rem;
		color: rgba(255, 255, 255, 0.5);
		font-weight: 300;
	}

	.status.error {
		color: #e74c3c;
		font-weight: 400;
	}

	.status.empty {
		color: rgba(255, 255, 255, 0.4);
	}

	.spinner {
		width: 40px;
		height: 40px;
		border: 3px solid rgba(255, 255, 255, 0.1);
		border-top-color: #3498db;
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	.history-container {
		display: flex;
		flex-direction: column-reverse;
		gap: 20px;
		padding-bottom: 40px;
	}

	.chat-entry {
		background: linear-gradient(135deg, rgba(44, 44, 44, 0.6) 0%, rgba(35, 35, 35, 0.6) 100%);
		border: 1px solid rgba(255, 255, 255, 0.05);
		border-radius: 12px;
		padding: 24px;
		display: flex;
		flex-direction: column;
		gap: 16px;
		backdrop-filter: blur(10px);
		transition: all 0.3s ease;
		position: relative;
	}

	.delete-btn {
		position: absolute;
		top: 12px;
		right: 12px;
		width: 28px;
		height: 28px;
		border: 1px solid rgba(255, 255, 255, 0.1);
		background: rgba(0, 0, 0, 0.3);
		color: rgba(255, 255, 255, 0.4);
		border-radius: 6px;
		font-size: 1.5rem;
		line-height: 1;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		opacity: 0;
		transition: all 0.2s ease;
		padding: 0;
	}

	.chat-entry:hover .delete-btn {
		opacity: 1;
	}

	.delete-btn:hover {
		background: rgba(231, 76, 60, 0.2);
		border-color: rgba(231, 76, 60, 0.4);
		color: #e74c3c;
		transform: scale(1.1);
	}

	.delete-btn:active {
		transform: scale(0.95);
	}

	.chat-entry:hover {
		border-color: rgba(255, 255, 255, 0.1);
		transform: translateY(-2px);
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
	}

	.message-part {
		display: flex;
		flex-direction: column;
		gap: 8px;
		padding: 12px 0;
	}

	.message-label {
		font-size: 0.75rem;
		font-weight: 600;
		letter-spacing: 1px;
		opacity: 0.7;
		text-transform: uppercase;
	}

	.message-content {
		line-height: 1.7;
		white-space: pre-wrap;
		font-size: 0.95rem;
		font-weight: 300;
	}

	.message-user .message-label {
		color: #a9d1ff;
	}

	.message-user .message-content {
		color: #cce0ff;
	}

	.message-murph .message-label {
		color: #a9ffb0;
	}

	.message-murph .message-content {
		color: #d1ffd4;
	}

	.message-system {
		font-style: italic;
		opacity: 0.6;
	}

	.message-system .message-label {
		color: #f3ff80;
	}

	.message-system .message-content {
		color: #faffbd;
		font-size: 0.85rem;
	}

	@media (max-width: 768px) {
		main {
			padding: 40px 20px 30px;
		}

		h1 {
			font-size: 2rem;
		}

		.chat-entry {
			padding: 20px;
		}
	}
</style>
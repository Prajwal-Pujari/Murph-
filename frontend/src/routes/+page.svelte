<script lang="ts">
	import { onMount } from 'svelte';
	import Clock from '$lib/components/Clock.svelte';
	import { now, equal } from '$lib/utilities/time';
	import { goto } from '$app/navigation'; // <-- ADDED: SvelteKit's navigation function

	let time = now();
	let showMurph = false;
	let isRecording = false;
	let mediaRecorder: MediaRecorder | null = null;
	let audioChunks: Blob[] = [];

	onMount(() => {
		const clockInterval = setInterval(() => {
			if (!showMurph) {
				const newTime = now();
				if (!equal(newTime, time)) {
					time = newTime;
				}
			}
		}, 100);

		// --- MODIFIED: handleKeyDown function ---
		const handleKeyDown = (event: KeyboardEvent) => {
			// ADDED: Shortcut for history page
			if (event.ctrlKey && event.shiftKey && event.code === 'KeyH') {
				event.preventDefault();
				goto('/history');
				return; // Stop processing this event
			}

			// Existing logic for spacebar
			if (event.code === 'Space' && !isRecording) {
				event.preventDefault();
				startRecording();
			}
		};

		const handleKeyUp = (event: KeyboardEvent) => {
			if (event.code === 'Space' && isRecording) {
				event.preventDefault();
				stopRecording();
			}
		};

		window.addEventListener('keydown', handleKeyDown);
		window.addEventListener('keyup', handleKeyUp);

		return () => {
			clearInterval(clockInterval);
			window.removeEventListener('keydown', handleKeyDown);
			window.removeEventListener('keyup', handleKeyUp);
		};
	});

	async function startRecording() {
		try {
			showMurph = true;
			time = { hours: 'MU', minutes: 'RP', seconds: 'H!' };

			const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
			isRecording = true;
			audioChunks = [];

			mediaRecorder = new MediaRecorder(stream);

			mediaRecorder.ondataavailable = (event: BlobEvent) => {
				audioChunks.push(event.data);
			};

			mediaRecorder.onstop = async () => {
				const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
				const formData = new FormData();
				formData.append('audio', audioBlob, 'recording.webm');

				try {
					const response = await fetch('http://localhost:8000/voice-chat', {
						method: 'POST',
						body: formData
					});

					if (response.ok) {
						const responseAudioBlob = await response.blob();
						const audioUrl = URL.createObjectURL(responseAudioBlob);
						const audio = new Audio(audioUrl);
						audio.play();

						audio.onended = () => {
							URL.revokeObjectURL(audioUrl);
						};
					}
				} catch (err) {
					console.error('Failed to get AI voice response:', err);
				}

				showMurph = false;
			};

			mediaRecorder.start();
		} catch (err) {
			console.error('Error accessing microphone:', err);
			showMurph = false;
			isRecording = false;
		}
	}

	function stopRecording() {
		if (mediaRecorder && isRecording) {
			mediaRecorder.stop();
			isRecording = false;
		}
	}
</script>

<svelte:head>
	<title>Murph</title>
</svelte:head>

<div id="app" class:murph-active={showMurph}>
	<Clock {time} />
</div>

<div class="instructions">
	<p>Press and hold <kbd>SPACE</kbd> to speak</p>
	<p class="shortcut">
		Press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>H</kbd> for history
	</p>
</div>

<style>
	.instructions {
		position: fixed;
		bottom: 2rem;
		left: 50%;
		transform: translateX(-50%);
		text-align: center;
		color: oklch(60% 0 0);
	}

	/* ADDED: Style for the new shortcut text */
	.shortcut {
		font-size: 0.9em;
		color: oklch(50% 0 0);
		margin-top: 0.5rem;
	}
	/* ADDED: Make the shortcut kbd elements slightly smaller */
	.shortcut kbd {
		padding: 3px 8px;
		font-size: 0.85em;
	}

	kbd {
		background-color: oklch(30% 0 0);
		border: 1px solid oklch(40% 0 0);
		border-radius: 3px;
		box-shadow: 0 1px 0 rgba(0, 0, 0, 0.2);
		/* UPDATED: To use the variable for color change */
		color: var(--hand-color, oklch(94.75% 0.1916 119.15));
		transition: color 250ms ease-out;
		display: inline-block;
		font-family: monospace;
		font-size: 0.9em;
		padding: 4px 12px;
		white-space: nowrap;
	}
</style>
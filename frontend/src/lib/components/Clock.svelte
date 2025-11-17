<script lang="ts">
  import Field from './Field.svelte';
  import type { Time } from '$lib/types/time';
  
  export let time: Time;
</script>

<div class="clock">
  <Field field="hours" value={time.hours} />
  <Field field="minutes" value={time.minutes} />
  <Field field="seconds" value={time.seconds} />
</div>

<style>
  .clock {
    display: flex;
    gap: 1rem;
  }
</style>
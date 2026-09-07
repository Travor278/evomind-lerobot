'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

import type { RuntimeEvent } from './page';

const SOUND_SETTING_KEY = 'evomind-lerobot:sound';
const sounds = {
  enabled: '/audio/sound-enabled.wav',
  recordingStarted: '/audio/recording-started.wav',
  episodeSaving: '/audio/episode-saving.wav',
  episodeSaved: '/audio/episode-saved.wav',
  resetting: '/audio/resetting.wav',
  rerecording: '/audio/rerecording.wav',
  recordingCompleted: '/audio/recording-completed.wav',
  recordingFailed: '/audio/recording-failed.wav',
} as const;

type SoundName = keyof typeof sounds;

function isEpisodeStart(event: RuntimeEvent) {
  return event.operation === 'recording'
    && event.phase === 'running'
    && event.data.stage === 'episode';
}

function soundForEvent(
  event: RuntimeEvent,
  startedJobs: Set<string>,
  previousPhase: string | undefined,
): SoundName | null {
  if (event.operation !== 'recording') return null;

  const jobId = event.job_id;
  if (event.data.stage === 'episode_saved') return 'episodeSaved';
  if (event.phase === 'resetting' && previousPhase !== 'resetting' && event.data.rerecord_episode === true) return 'rerecording';
  if (isEpisodeStart(event)) {
    if (!jobId || startedJobs.has(jobId)) return null;
    startedJobs.add(jobId);
    return 'recordingStarted';
  }
  if (event.phase === 'saving') return 'episodeSaving';
  if (event.phase === 'resetting' && previousPhase !== 'resetting') return 'resetting';
  if (event.phase === 'completed') {
    if (jobId) startedJobs.delete(jobId);
    return 'recordingCompleted';
  }
  if (event.phase === 'failed') {
    if (jobId) startedJobs.delete(jobId);
    return 'recordingFailed';
  }
  return null;
}

export function useRuntimeSounds(event: RuntimeEvent | null) {
  const [enabled, setEnabled] = useState(true);
  const [ready, setReady] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const lastEvent = useRef<{ sequence: number; timestamp: number } | null>(null);
  const startedJobs = useRef(new Set<string>());
  const finishedJobs = useRef(new Set<string>());
  const phaseByJob = useRef(new Map<string, string>());

  const play = useCallback(async (name: SoundName) => {
    const audio = audioRef.current ?? new Audio();
    audioRef.current = audio;
    audio.pause();
    audio.src = sounds[name];
    audio.currentTime = 0;
    await audio.play();
  }, []);

  useEffect(() => {
    let timer: number | undefined;
    try {
      if (window.localStorage.getItem(SOUND_SETTING_KEY) === 'off') {
        timer = window.setTimeout(() => setEnabled(false), 0);
      }
    }
    catch { /* Sound remains enabled when browser storage is unavailable. */ }
    return () => {
      if (timer !== undefined) window.clearTimeout(timer);
      audioRef.current?.pause();
    };
  }, []);

  useEffect(() => {
    if (!event) return;
    const timestamp = Date.parse(event.timestamp) || 0;
    if (lastEvent.current === null) {
      lastEvent.current = { sequence: event.sequence, timestamp };
      phaseByJob.current.set(event.job_id ?? event.operation, event.phase);
      if (isEpisodeStart(event) && event.job_id) startedJobs.current.add(event.job_id);
      if ((event.phase === 'completed' || event.phase === 'failed') && event.job_id) finishedJobs.current.add(event.job_id);
      return;
    }
    if (event.sequence <= lastEvent.current.sequence && timestamp <= lastEvent.current.timestamp) return;
    lastEvent.current = { sequence: event.sequence, timestamp };
    const phaseKey = event.job_id ?? event.operation;
    const previousPhase = phaseByJob.current.get(phaseKey);
    phaseByJob.current.set(phaseKey, event.phase);

    if ((event.phase === 'completed' || event.phase === 'failed') && event.job_id) {
      if (finishedJobs.current.has(event.job_id)) return;
      finishedJobs.current.add(event.job_id);
    }

    const sound = soundForEvent(event, startedJobs.current, previousPhase);
    if (!enabled || !ready || !sound) return;
    void play(sound).catch(() => setReady(false));
  }, [enabled, event, play, ready]);

  const toggle = useCallback(async () => {
    if (enabled && ready) {
      audioRef.current?.pause();
      setEnabled(false);
      setReady(false);
      try { window.localStorage.setItem(SOUND_SETTING_KEY, 'off'); } catch { /* Keep the in-memory setting. */ }
      return;
    }

    setEnabled(true);
    try {
      await play('enabled');
      setReady(true);
      try { window.localStorage.setItem(SOUND_SETTING_KEY, 'on'); } catch { /* Keep the in-memory setting. */ }
    } catch {
      setReady(false);
    }
  }, [enabled, play, ready]);

  return {
    enabled,
    ready,
    label: enabled && ready ? '声音已开启' : '开启声音',
    toggle,
  };
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { findSegmentIndexByTime } from "../utils/timing.js";

function makeSessionKey(surahMeta) {
  const reciter = String(surahMeta?.reciter?.slug ?? "").trim();
  const surah = Number(surahMeta?.surah?.number);
  return `${reciter}:${Number.isFinite(surah) ? surah : ""}`;
}

function hasAyahAudio(row) {
  return typeof row?.audio_url === "string" && row.audio_url.trim();
}

function findNextAyahWithAudio(ayahs, startIndex) {
  for (let index = Math.max(0, startIndex); index < ayahs.length; index += 1) {
    if (hasAyahAudio(ayahs[index])) {
      return { row: ayahs[index], index };
    }
  }
  return null;
}

function findPrevAyahWithAudio(ayahs, startIndex) {
  for (let index = Math.min(ayahs.length - 1, startIndex); index >= 0; index -= 1) {
    if (hasAyahAudio(ayahs[index])) {
      return { row: ayahs[index], index };
    }
  }
  return null;
}

function currentAyahIndexFromState({ ayahs, activeAyah, selectedAyah, currentTime }) {
  let index = ayahs.findIndex((row) => row.ayah === activeAyah);
  if (index < 0) {
    index = ayahs.findIndex((row) => row.ayah === selectedAyah);
  }
  if (index < 0 && Number.isFinite(currentTime)) {
    index = ayahs.findIndex(
      (row) =>
        Number.isFinite(row?.start_s) &&
        Number.isFinite(row?.end_s) &&
        currentTime >= row.start_s - 0.05 &&
        currentTime <= row.end_s + 0.05,
    );
  }
  return index;
}

export function usePlaybackController({
  audioRef,
  resolvedAudioSrc,
  surahMeta,
  timingData,
  ayahs,
  words,
  setError,
}) {
  const segmentStopRef = useRef(null);
  const externalOffsetRef = useRef(0);
  const currentTimeRef = useRef(0);
  const activeAyahRef = useRef(null);
  const selectedAyahRef = useRef(null);
  const rafRef = useRef(null);
  const lastClockPaintRef = useRef(-1);

  const sessionKey = useMemo(() => makeSessionKey(surahMeta), [surahMeta]);

  const [manualAudioState, setManualAudioState] = useState({
    key: "",
    value: "",
  });
  const [activeAyahState, setActiveAyahState] = useState({
    key: "",
    value: null,
  });
  const [activeWordState, setActiveWordState] = useState({
    key: "",
    value: null,
  });
  const [selectedAyahState, setSelectedAyahState] = useState({
    key: "",
    value: null,
  });
  const [clockState, setClockState] = useState({ key: "", value: 0 });
  const [durationState, setDurationState] = useState({ key: "", value: 0 });
  const [playingState, setPlayingState] = useState({
    key: "",
    value: false,
  });
  const [pendingNextState, setPendingNextState] = useState({
    key: "",
    value: false,
  });

  const firstAyah = ayahs[0]?.ayah ?? null;
  const firstAyahWithAudio = useMemo(
    () => ayahs.find((entry) => hasAyahAudio(entry)) ?? null,
    [ayahs],
  );
  const firstAyahAudioSrc = firstAyahWithAudio?.audio_url?.trim() ?? "";

  const manualAudioSrc = manualAudioState.key === sessionKey ? manualAudioState.value : "";
  const hasContinuousAudio = Boolean(resolvedAudioSrc);
  const isAyahByAyah = !hasContinuousAudio && ayahs.some((entry) => hasAyahAudio(entry));
  const playerAudioSrc = hasContinuousAudio
    ? resolvedAudioSrc
    : manualAudioSrc || firstAyahAudioSrc;

  const activeAyah = activeAyahState.key === sessionKey ? activeAyahState.value : firstAyah;
  const selectedAyah = selectedAyahState.key === sessionKey ? selectedAyahState.value : firstAyah;
  const activeWordGlobal = activeWordState.key === sessionKey ? activeWordState.value : null;
  const currentTime = clockState.key === sessionKey ? clockState.value : 0;
  const durationS =
    durationState.key === sessionKey && Number.isFinite(durationState.value)
      ? durationState.value
      : (surahMeta?.audio?.duration_s ?? timingData?.audio?.duration_s ?? 0);
  const isPlaying = playingState.key === sessionKey ? playingState.value : false;
  const shouldHighlightNextAyah =
    pendingNextState.key === sessionKey ? pendingNextState.value : false;

  useEffect(() => {
    activeAyahRef.current = activeAyah;
    selectedAyahRef.current = selectedAyah;
  }, [activeAyah, selectedAyah]);

  const clearPendingNext = useCallback(() => {
    setPendingNextState((prev) => {
      if (prev.key === sessionKey && prev.value === false) {
        return prev;
      }
      return { key: sessionKey, value: false };
    });
  }, [sessionKey]);

  useEffect(() => {
    segmentStopRef.current = null;
    externalOffsetRef.current = Number.isFinite(firstAyahWithAudio?.start_s)
      ? firstAyahWithAudio.start_s
      : 0;
    currentTimeRef.current = 0;
    lastClockPaintRef.current = 0;

    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      audio.removeAttribute("src");
      audio.load();
    }
  }, [audioRef, firstAyahWithAudio?.start_s, sessionKey]);

  const wordsByAyah = useMemo(() => {
    const groups = new Map();
    for (const word of words) {
      if (!groups.has(word.ayah)) {
        groups.set(word.ayah, []);
      }
      groups.get(word.ayah).push(word);
    }
    return groups;
  }, [words]);

  const wordsByGlobal = useMemo(() => {
    const groups = new Map();
    for (const word of words) {
      groups.set(word.word_index_global, word);
    }
    return groups;
  }, [words]);

  const activeWord =
    activeWordGlobal !== null ? (wordsByGlobal.get(activeWordGlobal) ?? null) : null;
  const focusedAyah = activeWord?.ayah ?? selectedAyah ?? activeAyah;
  const selectedAyahWords = wordsByAyah.get(focusedAyah) ?? [];
  const displayAyahNumber = focusedAyah;
  const displayAyahWords = wordsByAyah.get(displayAyahNumber) ?? [];

  const syncFromTime = useCallback(
    (timeS) => {
      if (!Number.isFinite(timeS)) {
        return;
      }
      const timelineTimeS = timeS + externalOffsetRef.current;
      currentTimeRef.current = timelineTimeS;

      const audio = audioRef.current;
      if (
        audio &&
        segmentStopRef.current !== null &&
        timelineTimeS >= segmentStopRef.current - 0.02
      ) {
        audio.pause();
        segmentStopRef.current = null;
      }

      if (Math.abs(timelineTimeS - lastClockPaintRef.current) >= 0.05 || timelineTimeS === 0) {
        lastClockPaintRef.current = timelineTimeS;
        setClockState((prev) => {
          if (prev.key === sessionKey && prev.value === timelineTimeS) {
            return prev;
          }
          return { key: sessionKey, value: timelineTimeS };
        });
      }

      if (!ayahs.length && !words.length) {
        return;
      }

      const ayahIndex = findSegmentIndexByTime(ayahs, timelineTimeS, {
        tolerance: 0.03,
        holdInGap: true,
      });
      const nextAyah = ayahIndex >= 0 ? ayahs[ayahIndex].ayah : null;
      const fallbackAyah =
        ayahs.length > 0 && timelineTimeS < ayahs[0].start_s ? ayahs[0].ayah : null;
      const resolvedAyah = nextAyah ?? fallbackAyah;

      setActiveAyahState((prevAyah) => {
        if (prevAyah.key === sessionKey && prevAyah.value === resolvedAyah) {
          return prevAyah;
        }
        return { key: sessionKey, value: resolvedAyah };
      });

      if (resolvedAyah !== null) {
        setSelectedAyahState((prevAyah) => {
          if (prevAyah.key === sessionKey && prevAyah.value === resolvedAyah) {
            return prevAyah;
          }
          return { key: sessionKey, value: resolvedAyah };
        });
      }

      const wordIndex = findSegmentIndexByTime(words, timelineTimeS, {
        tolerance: 0,
        moveToNextInGap: true,
      });
      const nextWord = wordIndex >= 0 ? words[wordIndex].word_index_global : null;
      setActiveWordState((prevWord) => {
        if (prevWord.key === sessionKey && prevWord.value === nextWord) {
          return prevWord;
        }
        return { key: sessionKey, value: nextWord };
      });
    },
    [audioRef, ayahs, sessionKey, words],
  );

  const resolveCurrentAyahIndex = useCallback(() => {
    let index = ayahs.findIndex((row) => row.ayah === activeAyahRef.current);
    if (index < 0) {
      index = ayahs.findIndex((row) => row.ayah === selectedAyahRef.current);
    }
    if (index < 0) {
      const timeline = currentTimeRef.current;
      index = ayahs.findIndex(
        (row) =>
          Number.isFinite(row?.start_s) &&
          Number.isFinite(row?.end_s) &&
          timeline >= row.start_s - 0.05 &&
          timeline <= row.end_s + 0.05,
      );
    }
    return index;
  }, [ayahs]);

  const ensureContinuousSource = useCallback(
    (audio) => {
      if (!audio || !hasContinuousAudio || !resolvedAudioSrc) {
        return;
      }
      const nextSrc = resolvedAudioSrc.trim();
      if (!nextSrc) {
        return;
      }
      if (audio.src !== nextSrc) {
        audio.src = nextSrc;
      }
    },
    [hasContinuousAudio, resolvedAudioSrc],
  );

  const playExternalClip = useCallback(
    (url, { offsetS = 0, localStartS = 0, stopAtGlobalS = null } = {}) => {
      const audio = audioRef.current;
      if (!audio || typeof url !== "string" || !url.trim()) {
        return false;
      }

      clearPendingNext();
      externalOffsetRef.current = Number.isFinite(offsetS) ? Math.max(0, offsetS) : 0;
      setManualAudioState({ key: sessionKey, value: url.trim() });
      segmentStopRef.current = Number.isFinite(stopAtGlobalS) ? stopAtGlobalS : null;

      audio.src = url.trim();
      audio.currentTime = Number.isFinite(localStartS) ? Math.max(0, localStartS) : 0;
      syncFromTime(audio.currentTime);

      const playPromise = audio.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch((playError) => {
          if (playError?.name === "AbortError") {
            return;
          }
          setError(`Couldn't start playback (${String(playError)}).`);
        });
      }
      return true;
    },
    [audioRef, clearPendingNext, sessionKey, setError, syncFromTime],
  );

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return undefined;
    }

    const stopRaf = () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };

    const tick = () => {
      syncFromTime(audio.currentTime);
      if (!audio.paused && !audio.ended) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        rafRef.current = null;
      }
    };

    const startRaf = () => {
      stopRaf();
      rafRef.current = requestAnimationFrame(tick);
    };

    const onPlay = () => {
      clearPendingNext();
      setPlayingState({ key: sessionKey, value: true });
      startRaf();
    };

    const onPause = () => {
      setPlayingState({ key: sessionKey, value: false });
      syncFromTime(audio.currentTime);
      stopRaf();
    };

    const onEnded = () => {
      setPlayingState({ key: sessionKey, value: false });
      segmentStopRef.current = null;
      syncFromTime(audio.currentTime);
      stopRaf();

      if (isAyahByAyah) {
        const currentIndex = resolveCurrentAyahIndex();
        const nextPlayable = findNextAyahWithAudio(ayahs, currentIndex + 1);
        setPendingNextState({
          key: sessionKey,
          value: Boolean(nextPlayable),
        });
      } else {
        clearPendingNext();
      }
    };

    const onSeek = () => {
      syncFromTime(audio.currentTime);
    };

    const onLoadedMetadata = () => {
      const clipDuration = Number.isFinite(audio.duration) ? audio.duration : 0;
      const fallbackDuration = surahMeta?.audio?.duration_s ?? timingData?.audio?.duration_s ?? 0;
      const nextDuration = Number.isFinite(audio.duration)
        ? Math.max(fallbackDuration, externalOffsetRef.current + clipDuration)
        : fallbackDuration;

      setDurationState({ key: sessionKey, value: nextDuration });
      syncFromTime(audio.currentTime);
    };

    const onAudioError = () => {
      setError("Audio couldn't load for this selection. Check the source URL or network access.");
    };

    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);
    audio.addEventListener("seeking", onSeek);
    audio.addEventListener("seeked", onSeek);
    audio.addEventListener("timeupdate", onSeek);
    audio.addEventListener("loadedmetadata", onLoadedMetadata);
    audio.addEventListener("error", onAudioError);

    syncFromTime(audio.currentTime);

    return () => {
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
      audio.removeEventListener("seeking", onSeek);
      audio.removeEventListener("seeked", onSeek);
      audio.removeEventListener("timeupdate", onSeek);
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
      audio.removeEventListener("error", onAudioError);
      stopRaf();
    };
  }, [
    audioRef,
    ayahs,
    clearPendingNext,
    isAyahByAyah,
    resolveCurrentAyahIndex,
    sessionKey,
    setError,
    surahMeta?.audio?.duration_s,
    syncFromTime,
    timingData?.audio?.duration_s,
  ]);

  const playSegment = useCallback(
    (startS, endS) => {
      const audio = audioRef.current;
      if (!audio || !Number.isFinite(startS) || !Number.isFinite(endS)) {
        return;
      }

      clearPendingNext();
      externalOffsetRef.current = 0;
      segmentStopRef.current = endS;
      audio.currentTime = Math.max(0, startS);
      syncFromTime(audio.currentTime);

      const playPromise = audio.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch(() => {});
      }
    },
    [audioRef, clearPendingNext, syncFromTime],
  );

  const seekTo = useCallback(
    (timeS) => {
      const audio = audioRef.current;
      if (!audio || !Number.isFinite(timeS)) {
        return;
      }

      clearPendingNext();

      if (!hasContinuousAudio) {
        const targetAyah =
          ayahs.find(
            (row) =>
              Number.isFinite(row?.start_s) &&
              Number.isFinite(row?.end_s) &&
              timeS >= row.start_s &&
              timeS <= row.end_s,
          ) ?? firstAyahWithAudio;
        if (targetAyah && hasAyahAudio(targetAyah)) {
          const localStartS = Number.isFinite(targetAyah.start_s)
            ? Math.max(0, timeS - targetAyah.start_s)
            : 0;
          playExternalClip(targetAyah.audio_url, {
            offsetS: targetAyah.start_s,
            localStartS,
          });
        }
        return;
      }

      ensureContinuousSource(audio);
      externalOffsetRef.current = 0;
      segmentStopRef.current = null;
      audio.currentTime = Math.max(0, timeS);
      syncFromTime(audio.currentTime);
    },
    [
      audioRef,
      ayahs,
      clearPendingNext,
      ensureContinuousSource,
      firstAyahWithAudio,
      hasContinuousAudio,
      playExternalClip,
      syncFromTime,
    ],
  );

  const playAyah = useCallback(
    (ayahEntry) => {
      if (!ayahEntry) {
        return;
      }
      setSelectedAyahState({ key: sessionKey, value: ayahEntry.ayah });
      setActiveAyahState({ key: sessionKey, value: ayahEntry.ayah });
      setActiveWordState({ key: sessionKey, value: null });

      if (!hasContinuousAudio && hasAyahAudio(ayahEntry)) {
        playExternalClip(ayahEntry.audio_url, {
          offsetS: ayahEntry.start_s,
          localStartS: 0,
        });
        return;
      }

      playSegment(ayahEntry.start_s, ayahEntry.end_s);
    },
    [hasContinuousAudio, playExternalClip, playSegment, sessionKey],
  );

  const playWord = useCallback(
    (wordEntry) => {
      if (!wordEntry) {
        return;
      }
      setSelectedAyahState({ key: sessionKey, value: wordEntry.ayah });
      setActiveAyahState({ key: sessionKey, value: wordEntry.ayah });
      setActiveWordState({ key: sessionKey, value: wordEntry.word_index_global });

      if (!hasContinuousAudio) {
        const ayahMatch = ayahs.find((entry) => entry.ayah === wordEntry.ayah);
        if (ayahMatch && hasAyahAudio(ayahMatch)) {
          const localStartS =
            Number.isFinite(wordEntry.start_s) && Number.isFinite(ayahMatch.start_s)
              ? Math.max(0, wordEntry.start_s - ayahMatch.start_s)
              : 0;
          playExternalClip(ayahMatch.audio_url, {
            offsetS: ayahMatch.start_s,
            localStartS,
            stopAtGlobalS: wordEntry.end_s,
          });
          return;
        }
      }

      playSegment(wordEntry.start_s, wordEntry.end_s);
    },
    [ayahs, hasContinuousAudio, playExternalClip, playSegment, sessionKey],
  );

  const playFullSurah = useCallback(() => {
    clearPendingNext();

    if (!hasContinuousAudio) {
      const first = firstAyahWithAudio ?? ayahs[0] ?? null;
      if (!first || !hasAyahAudio(first)) {
        return;
      }

      setSelectedAyahState({ key: sessionKey, value: first.ayah });
      setActiveAyahState({ key: sessionKey, value: first.ayah });
      setActiveWordState({ key: sessionKey, value: null });
      playExternalClip(first.audio_url, {
        offsetS: first.start_s,
        localStartS: 0,
      });
      return;
    }

    const audio = audioRef.current;
    if (!audio) {
      return;
    }

    ensureContinuousSource(audio);
    externalOffsetRef.current = 0;
    segmentStopRef.current = null;
    audio.currentTime = 0;
    syncFromTime(0);

    const playPromise = audio.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {});
    }
  }, [
    audioRef,
    ayahs,
    clearPendingNext,
    ensureContinuousSource,
    firstAyahWithAudio,
    hasContinuousAudio,
    playExternalClip,
    sessionKey,
    syncFromTime,
  ]);

  const stopSegment = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    clearPendingNext();
    audio.pause();
    segmentStopRef.current = null;
    syncFromTime(audio.currentTime);
  }, [audioRef, clearPendingNext, syncFromTime]);

  const pausePlayback = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    clearPendingNext();
    audio.pause();
    syncFromTime(audio.currentTime);
  }, [audioRef, clearPendingNext, syncFromTime]);

  const resumePlayback = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }

    clearPendingNext();

    if (!audio.src) {
      playFullSurah();
      return;
    }

    ensureContinuousSource(audio);

    const playPromise = audio.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch((playError) => {
        setError(`Couldn't start playback (${String(playError)}).`);
      });
    }
  }, [audioRef, clearPendingNext, ensureContinuousSource, playFullSurah, setError]);

  const togglePlayback = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }

    if (audio.paused || audio.ended) {
      resumePlayback();
      return;
    }
    pausePlayback();
  }, [audioRef, pausePlayback, resumePlayback]);

  const currentAyahIndex = useMemo(
    () =>
      currentAyahIndexFromState({
        ayahs,
        activeAyah,
        selectedAyah,
        currentTime,
      }),
    [activeAyah, ayahs, currentTime, selectedAyah],
  );

  const prevAyahForNav = useMemo(
    () => findPrevAyahWithAudio(ayahs, currentAyahIndex - 1),
    [ayahs, currentAyahIndex],
  );
  const nextAyahForNav = useMemo(
    () => findNextAyahWithAudio(ayahs, currentAyahIndex + 1),
    [ayahs, currentAyahIndex],
  );

  const hasPrevAyah = Boolean(prevAyahForNav);
  const hasNextAyah = Boolean(nextAyahForNav);

  const playPrevAyah = useCallback(() => {
    if (!prevAyahForNav) {
      return;
    }
    playAyah(prevAyahForNav.row);
  }, [playAyah, prevAyahForNav]);

  const playNextAyah = useCallback(() => {
    if (!nextAyahForNav) {
      return;
    }
    playAyah(nextAyahForNav.row);
  }, [nextAyahForNav, playAyah]);

  const playActiveAyah = useCallback(() => {
    if (!ayahs.length) {
      return;
    }
    const match = ayahs.find((ayah) => ayah.ayah === activeAyah) ?? ayahs[0];
    playAyah(match);
  }, [activeAyah, ayahs, playAyah]);

  const playActiveWord = useCallback(() => {
    if (!words.length) {
      return;
    }
    const match = words.find((word) => word.word_index_global === activeWordGlobal) ?? words[0];
    playWord(match);
  }, [activeWordGlobal, playWord, words]);

  return {
    playerAudioSrc,
    activeAyah,
    activeWordGlobal,
    activeWord,
    focusedAyah,
    selectedAyahWords,
    displayAyahNumber,
    displayAyahWords,
    currentTime,
    durationS,
    isPlaying,
    isAyahByAyah,
    hasPrevAyah,
    hasNextAyah,
    shouldHighlightNextAyah,
    seekTo,
    playAyah,
    playWord,
    playFullSurah,
    playActiveAyah,
    playActiveWord,
    playPrevAyah,
    playNextAyah,
    pausePlayback,
    resumePlayback,
    togglePlayback,
    stopSegment,
  };
}

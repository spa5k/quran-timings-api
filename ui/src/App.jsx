import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const TIMING_EPSILON_S = 0.03

function formatStamp(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
  const mins = Math.floor(safe / 60)
  const secs = Math.floor(safe % 60)
  const millis = Math.floor((safe - Math.floor(safe)) * 1000)
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(millis).padStart(3, '0')}`
}

function findSegmentIndexByTime(
  segments,
  timeS,
  { tolerance = TIMING_EPSILON_S, holdInGap = false, moveToNextInGap = false } = {},
) {
  if (!Array.isArray(segments) || segments.length === 0 || !Number.isFinite(timeS)) {
    return -1
  }

  let left = 0
  let right = segments.length - 1
  let candidate = -1
  const target = timeS + tolerance

  while (left <= right) {
    const mid = (left + right) >> 1
    if (segments[mid].start_s <= target) {
      candidate = mid
      left = mid + 1
    } else {
      right = mid - 1
    }
  }

  if (candidate < 0) {
    return -1
  }

  const current = segments[candidate]
  if (timeS >= current.start_s - tolerance && timeS <= current.end_s + tolerance) {
    return candidate
  }

  const next = segments[candidate + 1]
  if (timeS > current.end_s && (!next || timeS < next.start_s)) {
    if (moveToNextInGap) {
      return next ? candidate + 1 : candidate
    }
    if (holdInGap) {
      return candidate
    }
  }

  return -1
}

function ensureVisible(container, item, smooth = false) {
  if (!container || !item) {
    return
  }

  const containerBounds = container.getBoundingClientRect()
  const itemBounds = item.getBoundingClientRect()
  const margin = 28

  const isInside =
    itemBounds.top >= containerBounds.top + margin && itemBounds.bottom <= containerBounds.bottom - margin

  if (!isInside) {
    item.scrollIntoView({
      block: 'center',
      behavior: smooth ? 'smooth' : 'auto',
    })
  }
}

function App() {
  const audioRef = useRef(null)
  const segmentStopRef = useRef(null)
  const rafRef = useRef(null)
  const lastClockPaintRef = useRef(-1)
  const ayahListRef = useRef(null)
  const wordListRef = useRef(null)
  const ayahRowRefs = useRef(new Map())
  const wordRowRefs = useRef(new Map())

  const [catalog, setCatalog] = useState(null)
  const [selectedReciterId, setSelectedReciterId] = useState('')
  const [selectedSurah, setSelectedSurah] = useState('')
  const [timingData, setTimingData] = useState(null)

  const [activeAyah, setActiveAyah] = useState(null)
  const [activeWordGlobal, setActiveWordGlobal] = useState(null)
  const [selectedAyah, setSelectedAyah] = useState(null)

  const [currentTime, setCurrentTime] = useState(0)
  const [durationS, setDurationS] = useState(0)
  const [isCatalogLoading, setIsCatalogLoading] = useState(true)
  const [isTimingLoading, setIsTimingLoading] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()

    const loadCatalog = async () => {
      try {
        setIsCatalogLoading(true)
        const response = await fetch('/data/catalog.json', { signal: controller.signal })
        if (!response.ok) {
          throw new Error(`catalog fetch failed (${response.status})`)
        }

        const loaded = await response.json()
        setCatalog(loaded)

        const firstRecitation = loaded?.recitations?.[0]
        const firstSurah = firstRecitation?.surahs?.[0]

        if (firstRecitation) {
          setSelectedReciterId(firstRecitation.id)
        }
        if (firstSurah) {
          setSelectedSurah(String(firstSurah.surah))
        }
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(`Failed to load catalog: ${String(loadError)}`)
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsCatalogLoading(false)
        }
      }
    }

    loadCatalog()
    return () => {
      controller.abort()
    }
  }, [])

  const recitations = useMemo(() => catalog?.recitations ?? [], [catalog])

  const reciter = useMemo(
    () => recitations.find((entry) => entry.id === selectedReciterId) ?? recitations[0] ?? null,
    [recitations, selectedReciterId],
  )

  const surahs = useMemo(() => reciter?.surahs ?? [], [reciter])

  const surahConfig = useMemo(
    () => surahs.find((entry) => String(entry.surah) === String(selectedSurah)) ?? surahs[0] ?? null,
    [selectedSurah, surahs],
  )

  useEffect(() => {
    const controller = new AbortController()

    const loadTiming = async () => {
      if (!surahConfig?.timingJson) {
        setTimingData(null)
        setDurationS(0)
        setSelectedAyah(null)
        setActiveAyah(null)
        setActiveWordGlobal(null)
        return
      }

      try {
        setError('')
        setIsTimingLoading(true)
        const response = await fetch(surahConfig.timingJson, { signal: controller.signal })
        if (!response.ok) {
          throw new Error(`timing fetch failed (${response.status})`)
        }

        const loaded = await response.json()
        if (controller.signal.aborted) {
          return
        }

        setTimingData(loaded)
        setDurationS(loaded?.audio?.duration_s ?? 0)

        const firstAyah = loaded?.ayahs?.[0]?.ayah ?? null
        setSelectedAyah(firstAyah)
        setActiveAyah(firstAyah)
        setActiveWordGlobal(null)
        setCurrentTime(0)
        lastClockPaintRef.current = 0
        setIsPlaying(false)

        if (audioRef.current) {
          audioRef.current.currentTime = 0
          audioRef.current.pause()
        }
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(`Failed to load timing data: ${String(loadError)}`)
          setTimingData(null)
          setDurationS(0)
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsTimingLoading(false)
        }
      }
    }

    loadTiming()
    return () => {
      controller.abort()
    }
  }, [surahConfig])

  const ayahs = useMemo(() => timingData?.ayahs ?? [], [timingData])
  const words = useMemo(() => timingData?.words ?? [], [timingData])

  const syncFromTime = useCallback(
    (timeS) => {
      if (!Number.isFinite(timeS)) {
        return
      }

      const audio = audioRef.current
      if (audio && segmentStopRef.current !== null && timeS >= segmentStopRef.current - 0.02) {
        audio.pause()
        segmentStopRef.current = null
      }

      if (Math.abs(timeS - lastClockPaintRef.current) >= 0.05 || timeS === 0) {
        lastClockPaintRef.current = timeS
        setCurrentTime(timeS)
      }

      if (!ayahs.length && !words.length) {
        return
      }

      const ayahIndex = findSegmentIndexByTime(ayahs, timeS, {
        tolerance: TIMING_EPSILON_S,
        holdInGap: true,
      })

      const nextAyah = ayahIndex >= 0 ? ayahs[ayahIndex].ayah : null
      setActiveAyah((prevAyah) => (prevAyah === nextAyah ? prevAyah : nextAyah))

      if (nextAyah !== null) {
        setSelectedAyah((prevAyah) => (prevAyah === nextAyah ? prevAyah : nextAyah))
      }

      const wordIndex = findSegmentIndexByTime(words, timeS, {
        tolerance: 0,
        moveToNextInGap: true,
      })
      const nextWord = wordIndex >= 0 ? words[wordIndex].word_index_global : null
      setActiveWordGlobal((prevWord) => (prevWord === nextWord ? prevWord : nextWord))
    },
    [ayahs, words],
  )

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) {
      return undefined
    }

    const stopRaf = () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
    }

    const tick = () => {
      syncFromTime(audio.currentTime)
      if (!audio.paused && !audio.ended) {
        rafRef.current = requestAnimationFrame(tick)
      } else {
        rafRef.current = null
      }
    }

    const startRaf = () => {
      stopRaf()
      rafRef.current = requestAnimationFrame(tick)
    }

    const onPlay = () => {
      setIsPlaying(true)
      startRaf()
    }

    const onPause = () => {
      setIsPlaying(false)
      syncFromTime(audio.currentTime)
      stopRaf()
    }

    const onEnded = () => {
      setIsPlaying(false)
      segmentStopRef.current = null
      syncFromTime(audio.currentTime)
      stopRaf()
    }

    const onSeek = () => {
      syncFromTime(audio.currentTime)
    }

    const onLoadedMetadata = () => {
      const nextDuration = Number.isFinite(audio.duration) ? audio.duration : timingData?.audio?.duration_s ?? 0
      setDurationS(nextDuration)
      syncFromTime(audio.currentTime)
    }

    const onAudioError = () => {
      setError('Audio failed to load. Please check network access for the audio source.')
    }

    audio.addEventListener('play', onPlay)
    audio.addEventListener('pause', onPause)
    audio.addEventListener('ended', onEnded)
    audio.addEventListener('seeking', onSeek)
    audio.addEventListener('seeked', onSeek)
    audio.addEventListener('timeupdate', onSeek)
    audio.addEventListener('loadedmetadata', onLoadedMetadata)
    audio.addEventListener('error', onAudioError)

    syncFromTime(audio.currentTime)

    return () => {
      audio.removeEventListener('play', onPlay)
      audio.removeEventListener('pause', onPause)
      audio.removeEventListener('ended', onEnded)
      audio.removeEventListener('seeking', onSeek)
      audio.removeEventListener('seeked', onSeek)
      audio.removeEventListener('timeupdate', onSeek)
      audio.removeEventListener('loadedmetadata', onLoadedMetadata)
      audio.removeEventListener('error', onAudioError)
      stopRaf()
    }
  }, [surahConfig?.audioSrc, syncFromTime, timingData?.audio?.duration_s])

  useEffect(() => {
    if (activeAyah === null) {
      return
    }
    const row = ayahRowRefs.current.get(activeAyah)
    ensureVisible(ayahListRef.current, row, true)
  }, [activeAyah])

  useEffect(() => {
    if (activeWordGlobal === null) {
      return
    }
    const row = wordRowRefs.current.get(activeWordGlobal)
    ensureVisible(wordListRef.current, row)
  }, [activeWordGlobal, selectedAyah])

  const wordsByAyah = useMemo(() => {
    const groups = new Map()
    if (!words.length) {
      return groups
    }

    for (const word of words) {
      if (!groups.has(word.ayah)) {
        groups.set(word.ayah, [])
      }
      groups.get(word.ayah).push(word)
    }

    return groups
  }, [words])

  const wordsByGlobal = useMemo(() => {
    const groups = new Map()
    for (const word of words) {
      groups.set(word.word_index_global, word)
    }
    return groups
  }, [words])

  const activeWord = activeWordGlobal !== null ? wordsByGlobal.get(activeWordGlobal) ?? null : null
  const focusedAyah = activeWord?.ayah ?? selectedAyah ?? activeAyah
  const selectedAyahWords = wordsByAyah.get(focusedAyah) ?? []
  const displayAyahNumber = focusedAyah
  const displayAyahWords = wordsByAyah.get(displayAyahNumber) ?? []

  const playSegment = useCallback(
    (startS, endS) => {
      const audio = audioRef.current
      if (!audio || !Number.isFinite(startS) || !Number.isFinite(endS)) {
        return
      }

      segmentStopRef.current = endS
      audio.currentTime = Math.max(0, startS)
      syncFromTime(audio.currentTime)

      const playPromise = audio.play()
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch(() => {})
      }
    },
    [syncFromTime],
  )

  const playFullSurah = useCallback(() => {
    const audio = audioRef.current
    if (!audio) {
      return
    }

    segmentStopRef.current = null
    audio.currentTime = 0
    syncFromTime(0)

    const playPromise = audio.play()
    if (playPromise && typeof playPromise.catch === 'function') {
      playPromise.catch(() => {})
    }
  }, [syncFromTime])

  const stopSegment = useCallback(() => {
    const audio = audioRef.current
    if (!audio) {
      return
    }
    audio.pause()
    segmentStopRef.current = null
    syncFromTime(audio.currentTime)
  }, [syncFromTime])

  const playAyah = useCallback((ayahEntry) => {
    if (!ayahEntry) {
      return
    }
    setSelectedAyah(ayahEntry.ayah)
    setActiveAyah(ayahEntry.ayah)
    playSegment(ayahEntry.start_s, ayahEntry.end_s)
  }, [playSegment])

  const playWord = useCallback((wordEntry) => {
    if (!wordEntry) {
      return
    }
    setSelectedAyah(wordEntry.ayah)
    setActiveAyah(wordEntry.ayah)
    setActiveWordGlobal(wordEntry.word_index_global)
    playSegment(wordEntry.start_s, wordEntry.end_s)
  }, [playSegment])

  const playActiveAyah = useCallback(() => {
    if (!ayahs.length) {
      return
    }

    const match = ayahs.find((ayah) => ayah.ayah === activeAyah) ?? ayahs[0]
    playAyah(match)
  }, [activeAyah, ayahs, playAyah])

  const playActiveWord = useCallback(() => {
    if (!words.length) {
      return
    }

    const match = words.find((word) => word.word_index_global === activeWordGlobal) ?? words[0]
    playWord(match)
  }, [activeWordGlobal, playWord, words])

  if (isCatalogLoading) {
    return <div className="status">Loading timing explorer...</div>
  }

  const currentSurahLabel = surahConfig
    ? `${surahConfig.surah}. ${surahConfig.title ?? `Surah ${surahConfig.surah}`}`
    : 'No surah selected'
  const progressPct = durationS > 0 ? Math.min(100, (currentTime / durationS) * 100) : 0
  const hasTimingData = ayahs.length > 0 || words.length > 0

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="kicker">Quran Audio Data</p>
        <h1>Timing Explorer</h1>
        <p className="subtitle">Recitation to Surah to Ayah by Ayah / Word by Word</p>
        <div className="hero-meta" aria-live="polite">
          <span className="pill">Surah {currentSurahLabel}</span>
          <span className={`pill ${isPlaying ? 'play-state-active' : ''}`}>{isPlaying ? 'Playing' : 'Paused'}</span>
          <span className="pill">Active Ayah: {activeAyah ?? '—'}</span>
          <span className="pill arabic-pill">{activeWord?.text_uthmani ?? '—'}</span>
        </div>
        <section className="active-ayah-panel" aria-label="Active ayah text">
          <p className="active-ayah-panel-label">Active Ayah Text {displayAyahNumber ? `(Ayah ${displayAyahNumber})` : ''}</p>
          {displayAyahWords.length ? (
            <p className="active-ayah-text" lang="ar" dir="rtl">
              {displayAyahWords.map((word) => (
                <span
                  key={`active-line-word-${word.word_index_global}`}
                  className={`active-ayah-word ${
                    activeWordGlobal === word.word_index_global ? 'active-ayah-word-current' : ''
                  }`}
                >
                  {word.text_uthmani}
                </span>
              ))}
            </p>
          ) : (
            <p className="active-ayah-empty">No ayah text available for this timing set.</p>
          )}
        </section>
      </header>

      <section className="controls">
        <label htmlFor="reciter-select">
          Recitation
          <select
            id="reciter-select"
            value={reciter?.id ?? ''}
            onChange={(event) => {
              const nextId = event.target.value
              setSelectedReciterId(nextId)
              const nextReciter = recitations.find((entry) => entry.id === nextId)
              const firstSurah = nextReciter?.surahs?.[0]
              setSelectedSurah(firstSurah ? String(firstSurah.surah) : '')
            }}
          >
            {recitations.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.name}
              </option>
            ))}
          </select>
        </label>

        <label htmlFor="surah-select">
          Surah
          <select
            id="surah-select"
            value={surahConfig ? String(surahConfig.surah) : ''}
            onChange={(event) => setSelectedSurah(event.target.value)}
          >
            {surahs.map((entry) => (
              <option key={entry.surah} value={entry.surah}>
                {entry.surah}. {entry.title ?? `Surah ${entry.surah}`}
              </option>
            ))}
          </select>
        </label>

        <div className="stat-grid">
          <div>
            <strong>{timingData?.ayahs?.length ?? 0}</strong>
            <span>Ayahs</span>
          </div>
          <div>
            <strong>{timingData?.words?.length ?? 0}</strong>
            <span>Words</span>
          </div>
          <div>
            <strong>{formatStamp(durationS)}</strong>
            <span>Duration</span>
          </div>
        </div>
      </section>

      <section className="player-panel">
        <div className="player-headline">
          <h2>Playback</h2>
          <p>{isTimingLoading ? 'Refreshing timing data…' : `At ${formatStamp(currentTime)}`}</p>
        </div>
        <audio
          key={surahConfig?.audioSrc ?? 'audio'}
          ref={audioRef}
          controls
          preload="metadata"
          src={surahConfig?.audioSrc ?? ''}
        />
        <div className="timeline" aria-hidden={!hasTimingData}>
          <div
            className="timeline-track"
            role="progressbar"
            aria-label="Playback progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(progressPct)}
          >
            <span className="timeline-fill" style={{ width: `${progressPct}%` }} />
          </div>
          <div className="timeline-marks">
            <span>{formatStamp(currentTime)}</span>
            <span>{formatStamp(durationS)}</span>
          </div>
        </div>
        <div className="player-actions">
          <button type="button" onClick={playFullSurah} disabled={!surahConfig?.audioSrc}>
            Play Full Surah
          </button>
          <button type="button" onClick={playActiveAyah} disabled={!ayahs.length}>
            Play Active Ayah
          </button>
          <button type="button" onClick={playActiveWord} disabled={!words.length}>
            Play Active Word
          </button>
          <button type="button" className="ghost" onClick={stopSegment} disabled={!hasTimingData}>
            Stop Segment
          </button>
        </div>
      </section>

      {error ? <p className="error">{error}</p> : null}

      <main className="content-grid">
        <section className="panel">
          <h2>Ayah Timeline</h2>
          <div ref={ayahListRef} className="list-scroll">
            {ayahs.map((ayah) => (
              <button
                type="button"
                key={`ayah-${ayah.ayah}`}
                ref={(node) => {
                  if (node) {
                    ayahRowRefs.current.set(ayah.ayah, node)
                  } else {
                    ayahRowRefs.current.delete(ayah.ayah)
                  }
                }}
                className={`segment-row ${activeAyah === ayah.ayah ? 'active' : ''} ${
                  focusedAyah === ayah.ayah ? 'selected' : ''
                }`}
                aria-current={activeAyah === ayah.ayah ? 'true' : undefined}
                onClick={() => playAyah(ayah)}
              >
                <span className="segment-title">Ayah {ayah.ayah}</span>
                <span className="segment-time">
                  {formatStamp(ayah.start_s)} {'→'} {formatStamp(ayah.end_s)}
                </span>
              </button>
            ))}
            {!ayahs.length ? <p className="empty-state">No ayah timings available.</p> : null}
          </div>
        </section>

        <section className="panel">
          <h2>Words {focusedAyah ? `(Ayah ${focusedAyah})` : ''}</h2>
          <div ref={wordListRef} className="list-scroll words">
            {selectedAyahWords.map((word) => (
              <button
                type="button"
                key={`word-${word.word_index_global}`}
                ref={(node) => {
                  if (node) {
                    wordRowRefs.current.set(word.word_index_global, node)
                  } else {
                    wordRowRefs.current.delete(word.word_index_global)
                  }
                }}
                className={`segment-row ${activeWordGlobal === word.word_index_global ? 'active-word' : ''}`}
                aria-current={activeWordGlobal === word.word_index_global ? 'true' : undefined}
                onClick={() => playWord(word)}
              >
                <span className="segment-title arabic">{word.text_uthmani}</span>
                <span className="segment-time">
                  {formatStamp(word.start_s)} {'→'} {formatStamp(word.end_s)}
                </span>
              </button>
            ))}
            {!selectedAyahWords.length ? <p className="empty-state">No word timings for this ayah.</p> : null}
          </div>
        </section>
      </main>
    </div>
  )
}

export default App

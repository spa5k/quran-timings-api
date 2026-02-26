import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

function formatStamp(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
  const mins = Math.floor(safe / 60)
  const secs = Math.floor(safe % 60)
  const millis = Math.floor((safe - Math.floor(safe)) * 1000)
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(millis).padStart(3, '0')}`
}

function App() {
  const audioRef = useRef(null)
  const segmentStopRef = useRef(null)

  const [catalog, setCatalog] = useState(null)
  const [selectedReciterId, setSelectedReciterId] = useState('')
  const [selectedSurah, setSelectedSurah] = useState('')
  const [timingData, setTimingData] = useState(null)

  const [activeAyah, setActiveAyah] = useState(null)
  const [activeWordGlobal, setActiveWordGlobal] = useState(null)
  const [selectedAyah, setSelectedAyah] = useState(null)

  const [currentTime, setCurrentTime] = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const loadCatalog = async () => {
      try {
        setIsLoading(true)
        const response = await fetch('/data/catalog.json')
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
        setError(`Failed to load catalog: ${String(loadError)}`)
      } finally {
        setIsLoading(false)
      }
    }

    loadCatalog()
  }, [])

  const recitations = catalog?.recitations ?? []

  const reciter = useMemo(
    () => recitations.find((entry) => entry.id === selectedReciterId) ?? recitations[0] ?? null,
    [recitations, selectedReciterId],
  )

  const surahs = reciter?.surahs ?? []

  const surahConfig = useMemo(
    () => surahs.find((entry) => String(entry.surah) === String(selectedSurah)) ?? surahs[0] ?? null,
    [selectedSurah, surahs],
  )

  useEffect(() => {
    const loadTiming = async () => {
      if (!surahConfig?.timingJson) {
        setTimingData(null)
        return
      }

      try {
        setError('')
        const response = await fetch(surahConfig.timingJson)
        if (!response.ok) {
          throw new Error(`timing fetch failed (${response.status})`)
        }

        const loaded = await response.json()
        setTimingData(loaded)

        const firstAyah = loaded?.ayahs?.[0]?.ayah ?? null
        setSelectedAyah(firstAyah)
        setActiveAyah(firstAyah)
        setActiveWordGlobal(null)
        setCurrentTime(0)

        if (audioRef.current) {
          audioRef.current.currentTime = 0
          audioRef.current.pause()
        }
      } catch (loadError) {
        setError(`Failed to load timing data: ${String(loadError)}`)
        setTimingData(null)
      }
    }

    loadTiming()
  }, [surahConfig])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) {
      return undefined
    }

    const onTimeUpdate = () => {
      const t = audio.currentTime
      setCurrentTime(t)

      if (segmentStopRef.current !== null && t >= segmentStopRef.current - 0.02) {
        audio.pause()
        segmentStopRef.current = null
      }

      if (!timingData) {
        return
      }

      const ayah = timingData.ayahs?.find((entry) => t >= entry.start_s && t <= entry.end_s)
      const word = timingData.words?.find((entry) => t >= entry.start_s && t <= entry.end_s)

      if (ayah) {
        setActiveAyah(ayah.ayah)
      }
      if (word) {
        setActiveWordGlobal(word.word_index_global)
      }
    }

    audio.addEventListener('timeupdate', onTimeUpdate)
    return () => {
      audio.removeEventListener('timeupdate', onTimeUpdate)
    }
  }, [timingData])

  const wordsByAyah = useMemo(() => {
    const groups = new Map()
    if (!timingData?.words) {
      return groups
    }

    for (const word of timingData.words) {
      if (!groups.has(word.ayah)) {
        groups.set(word.ayah, [])
      }
      groups.get(word.ayah).push(word)
    }
    return groups
  }, [timingData])

  const selectedAyahWords = wordsByAyah.get(selectedAyah) ?? []

  const playSegment = (start, end) => {
    const audio = audioRef.current
    if (!audio) {
      return
    }

    segmentStopRef.current = end
    audio.currentTime = start
    void audio.play()
  }

  const stopSegment = () => {
    const audio = audioRef.current
    if (!audio) {
      return
    }
    segmentStopRef.current = null
    audio.pause()
  }

  const playAyah = (ayahEntry) => {
    setSelectedAyah(ayahEntry.ayah)
    setActiveAyah(ayahEntry.ayah)
    playSegment(ayahEntry.start_s, ayahEntry.end_s)
  }

  const playWord = (wordEntry) => {
    setSelectedAyah(wordEntry.ayah)
    setActiveAyah(wordEntry.ayah)
    setActiveWordGlobal(wordEntry.word_index_global)
    playSegment(wordEntry.start_s, wordEntry.end_s)
  }

  const playActiveAyah = () => {
    if (!timingData?.ayahs?.length) {
      return
    }

    const match = timingData.ayahs.find((ayah) => ayah.ayah === activeAyah) ?? timingData.ayahs[0]
    playAyah(match)
  }

  const playActiveWord = () => {
    if (!timingData?.words?.length) {
      return
    }

    const match =
      timingData.words.find((word) => word.word_index_global === activeWordGlobal) ?? timingData.words[0]
    playWord(match)
  }

  if (isLoading) {
    return <div className="status">Loading timing explorer...</div>
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="kicker">Quran Audio Data</p>
        <h1>Timing Explorer</h1>
        <p className="subtitle">Recitation to Surah to Ayah by Ayah / Word by Word</p>
      </header>

      <section className="controls">
        <label>
          Recitation
          <select
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

        <label>
          Surah
          <select
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
            <strong>{formatStamp(currentTime)}</strong>
            <span>Playhead</span>
          </div>
        </div>
      </section>

      <section className="player-panel">
        <audio
          key={surahConfig?.audioSrc ?? 'audio'}
          ref={audioRef}
          controls
          preload="metadata"
          src={surahConfig?.audioSrc ?? ''}
        />
        <div className="player-actions">
          <button type="button" onClick={playActiveAyah}>
            Play Active Ayah
          </button>
          <button type="button" onClick={playActiveWord}>
            Play Active Word
          </button>
          <button type="button" className="ghost" onClick={stopSegment}>
            Stop Segment
          </button>
        </div>
      </section>

      {error ? <p className="error">{error}</p> : null}

      <main className="content-grid">
        <section className="panel">
          <h2>Ayah Timeline</h2>
          <div className="list-scroll">
            {timingData?.ayahs?.map((ayah) => (
              <button
                type="button"
                key={`ayah-${ayah.ayah}`}
                className={`segment-row ${activeAyah === ayah.ayah ? 'active' : ''}`}
                onClick={() => playAyah(ayah)}
              >
                <span className="segment-title">Ayah {ayah.ayah}</span>
                <span className="segment-time">
                  {formatStamp(ayah.start_s)} {'→'} {formatStamp(ayah.end_s)}
                </span>
              </button>
            ))}
          </div>
        </section>

        <section className="panel">
          <h2>Words {selectedAyah ? `(Ayah ${selectedAyah})` : ''}</h2>
          <div className="list-scroll words">
            {selectedAyahWords.map((word) => (
              <button
                type="button"
                key={`word-${word.word_index_global}`}
                className={`segment-row ${activeWordGlobal === word.word_index_global ? 'active-word' : ''}`}
                onClick={() => playWord(word)}
              >
                <span className="segment-title arabic">{word.text_uthmani}</span>
                <span className="segment-time">
                  {formatStamp(word.start_s)} {'→'} {formatStamp(word.end_s)}
                </span>
              </button>
            ))}
          </div>
        </section>
      </main>
    </div>
  )
}

export default App

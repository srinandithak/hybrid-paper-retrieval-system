import { useState, useRef } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const LOADING_WORDS = [
  'cross-referencing',
  'fact-checking',
  'footnoting',
  'shelf-browsing',
  'citation-hunting',
  'peer-reviewing',
]

// Splits comparison text on [paper_id] citation markers and renders
// each one as a small catalog-stamp tag rather than plain brackets.
function CitedText({ text }) {
  const parts = text.split(/(\[[^\[\]]+\])/g)
  return (
    <p className="comparison-text">
      {parts.map((part, i) => {
        const match = part.match(/^\[([^\[\]]+)\]$/)
        if (match) {
          return (
            <span className="stamp" key={i}>
              {match[1]}
            </span>
          )
        }
        return <span key={i}>{part}</span>
      })}
    </p>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingWord, setLoadingWord] = useState(LOADING_WORDS[0])
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const intervalRef = useRef(null)

  const runCompare = async (e) => {
    e.preventDefault()
    if (!query.trim() || loading) return

    setLoading(true)
    setError(null)
    setResult(null)

    let i = 0
    setLoadingWord(LOADING_WORDS[0])
    intervalRef.current = setInterval(() => {
      i = (i + 1) % LOADING_WORDS.length
      setLoadingWord(LOADING_WORDS[i])
    }, 1400)

    try {
      const res = await fetch(
        `${API_URL}/compare?query=${encodeURIComponent(query)}&limit=8`
      )
      if (!res.ok) throw new Error(`Server responded with ${res.status}`)
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(
        'Could not reach the paper database right now. Try again in a moment.'
      )
    } finally {
      clearInterval(intervalRef.current)
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="header">
        <span className="eyebrow">Vol. I — Computer Vision</span>
        <h1>Paper Compare</h1>
        <p className="subhead">
          Ask a question about object detection, vision transformers, or CNNs.
          Every claim below is cited to a real, retrieved paper.
        </p>
      </header>

      <form className="search-form" onSubmit={runCompare}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. how do vision transformers compare to CNNs for object detection?"
          aria-label="Research question"
        />
        <button type="submit" disabled={loading}>
          {loading ? loadingWord : 'Compare'}
        </button>
      </form>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <section className="result">
          <div className="result-meta">
            <span className="query-label">Query</span>
            <span className="query-value">{result.query}</span>
          </div>

          <CitedText text={result.comparison} />

          {result.papers_used && result.papers_used.length > 0 && (
            <div className="sources">
              <span className="sources-label">Retrieved from</span>
              <div className="stamp-row">
                {result.papers_used.map((id) => (
                  <span className="stamp stamp-muted" key={id}>
                    {id}
                  </span>
                ))}
              </div>
            </div>
          )}

          {result.validation && (
            <div
              className={
                'validation-note ' +
                (result.validation.is_fully_grounded ? 'grounded' : 'flagged')
              }
            >
              {result.validation.is_fully_grounded
                ? 'Every citation above matches a retrieved paper.'
                : `Some citations could not be verified: ${result.validation.invalid_citations.join(', ')}`}
            </div>
          )}
        </section>
      )}
    </div>
  )
}
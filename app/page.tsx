'use client'

import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from '@/components/ui/button'

// ── Types ────────────────────────────────────────────────────────────────────

interface ApiResult {
  predicted_emotion: string
  confidence: number                        // float 0–1 from backend
  all_probabilities: Record<string, number> // e.g. { anger: 0.03, happy: 0.95, … }
}

interface DisplayResult {
  emotion: string
  confidence: number                        // 0–100 for the UI
  all_probabilities: Record<string, number> // kept as 0–1, converted when rendering
}

// ── Image-type mapping ───────────────────────────────────────────────────────

const IMAGE_TYPE_MAP: Record<'full-face' | 'cropped', string> = {
  'full-face': 'full_face',
  cropped:     'cropped_ocular',
}

// ── Main Component ───────────────────────────────────────────────────────────

export default function Home() {
  const [image,     setImage]     = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imageType, setImageType] = useState<'full-face' | 'cropped'>('full-face')
  const [isLoading, setIsLoading] = useState(false)
  const [result,    setResult]    = useState<DisplayResult | null>(null)
  const [error,     setError]     = useState<string | null>(null)
  const [isDragActive, setIsDragActive] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const resultsRef   = useRef<HTMLDivElement>(null)

  // ── File helpers ────────────────────────────────────────────────────────

  const loadFile = (file: File) => {
    setImageFile(file)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setImage(e.target?.result as string)
    reader.readAsDataURL(file)
  }

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(e.type === 'dragenter' || e.type === 'dragover')
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
    const file = e.dataTransfer.files?.[0]
    if (file) loadFile(file)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) loadFile(file)
  }

  // ── API call ────────────────────────────────────────────────────────────

  const handleAnalyze = async () => {
    if (!imageFile) return

    setIsLoading(true)
    setError(null)

    try {
      const form = new FormData()
      form.append('file',       imageFile)
      form.append('image_type', IMAGE_TYPE_MAP[imageType])

      const response = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        body: form,
      })

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody?.detail ?? `Server error: ${response.status}`)
      }

      const data: ApiResult = await response.json()

      setResult({
        emotion:           data.predicted_emotion,
        confidence:        Math.round(data.confidence * 100),
        all_probabilities: data.all_probabilities,
      })

      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' })
      }, 300)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred.')
    } finally {
      setIsLoading(false)
    }
  }

  // ── Reset ───────────────────────────────────────────────────────────────

  const handleReset = () => {
    setResult(null)
    setImage(null)
    setImageFile(null)
    setImageType('full-face')
    setError(null)
  }

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <main className="min-h-screen bg-zinc-900 text-white overflow-x-hidden">

      {/* ── Hero ── */}
      <section className="min-h-screen flex flex-col items-center justify-center px-6 py-20 relative">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="text-center max-w-2xl"
        >
          <h1 className="text-5xl md:text-6xl font-bold mb-6 tracking-tight">
            Ocular Emotion
            <br />
            Detector
          </h1>
          <p className="text-lg md:text-xl text-slate-300 leading-relaxed">
            ML Model to detect Emotion from Eye Region <br></br> Team Members: 
            <ul className='text-[#FFAAB8]'>
              <li>Vishesh Agarwal</li>
              <li>Priyakshi Sonowal</li>
              <li>Nyasha Singh</li>
            </ul>
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 0.6 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="text-[#FFD8DF]"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </motion.div>
        </motion.div>
      </section>

      {/* ── Upload & Analyze ── */}
      <section className="min-h-screen flex flex-col items-center justify-center px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true, margin: '-100px' }}
          className="w-full max-w-2xl"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-12 text-center">Upload & Analyze</h2>

          {/* Drop zone */}
          <motion.div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            animate={{
              borderColor:     isDragActive ? '#346739' : 'rgb(51, 65, 85)',
              backgroundColor: isDragActive ? 'rgba(52, 103, 57, 0.05)' : 'rgba(52, 103, 57, 0.01)',
            }}
            className="relative mb-10 p-12 border-2 border-dashed rounded-lg cursor-pointer transition-colors duration-200"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />

            {image ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex flex-col items-center gap-4"
              >
                <img src={image} alt="Preview" className="max-h-64 rounded-lg object-contain" />
                <p className="text-sm text-slate-400">Click to replace image</p>
              </motion.div>
            ) : (
              <div className="flex flex-col items-center gap-3 py-8">
                <svg className="w-12 h-12 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                </svg>
                <div className="text-center">
                  <p className="text-base font-medium">Drag and drop your image here</p>
                  <p className="text-sm text-slate-400">or click to browse</p>
                </div>
              </div>
            )}
          </motion.div>

          {/* Image type selector */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.6 }}
            viewport={{ once: true }}
            className="mb-10"
          >
            <label className="block text-sm font-medium mb-4 text-slate-300">
              What type of image is this?
            </label>
            <div className="grid grid-cols-2 gap-4">
              {([
                { id: 'full-face', label: 'Full Face Image' },
                { id: 'cropped',   label: 'Cropped Ocular Region' },
              ] as const).map((option) => (
                <motion.button
                  key={option.id}
                  onClick={() => setImageType(option.id)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`py-4 px-6 rounded-lg font-medium transition-all duration-200 ${
                    imageType === option.id
                      ? 'text-white'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                  style={
                    imageType === option.id
                      ? { backgroundColor: '#FFAAB8' }
                      : { backgroundColor: 'rgba(51, 65, 85, 0.5)' }
                  }
                >
                  {option.label}
                </motion.button>
              ))}
            </div>
          </motion.div>

          {/* Error banner */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                className="mb-6 px-5 py-4 rounded-lg bg-red-950 border border-red-700 text-red-300 text-sm"
              >
                <span className="font-semibold">Error: </span>{error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Submit */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.6 }}
            viewport={{ once: true }}
          >
            <Button
              onClick={handleAnalyze}
              disabled={!imageFile || isLoading}
              className="w-full py-6 text-lg font-semibold rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ backgroundColor: '#FFAAB8', color: 'white' }}
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                  />
                  Analyzing Emotion…
                </span>
              ) : (
                'Analyze Emotion'
              )}
            </Button>
          </motion.div>
        </motion.div>
      </section>

      {/* ── Results ── */}
      <AnimatePresence>
        {result && (
          <section
            ref={resultsRef}
            className="min-h-screen flex flex-col items-center justify-center px-6 py-20"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.6 }}
              className="text-center w-full max-w-2xl"
            >
              {/* Predicted emotion + confidence circle */}
              <motion.div
                initial={{ y: -20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.6 }}
                className="mb-8"
              >
                <p className="text-slate-400 text-sm tracking-widest uppercase mb-6">
                  Detection Result
                </p>
                <h2
                  className="text-7xl md:text-8xl font-bold mb-6 tracking-tight capitalize"
                  style={{ color: '#FFD8DF' }}
                >
                  {result.emotion}
                </h2>
              </motion.div>

              <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.4, duration: 0.6 }}
                className="mb-10"
              >
                <div className="inline-flex items-center gap-4 bg-slate-800 rounded-lg px-8 py-6">
                  <div className="flex-1 text-left">
                    <p className="text-slate-400 text-sm mb-2">Confidence Level</p>
                    <p className="text-3xl font-bold">{result.confidence}%</p>
                  </div>
                  <div className="w-32 h-32 relative">
                    <motion.svg
                      className="transform -rotate-90"
                      width="128"
                      height="128"
                      viewBox="0 0 128 128"
                    >
                      <circle
                        cx="64" cy="64" r="56"
                        fill="none"
                        stroke="rgba(51, 65, 85, 0.3)"
                        strokeWidth="8"
                      />
                      <motion.circle
                        cx="64" cy="64" r="56"
                        fill="none"
                        stroke="#FFAAB8"
                        strokeWidth="8"
                        strokeDasharray={`${2 * Math.PI * 56}`}
                        initial={{ strokeDashoffset: 2 * Math.PI * 56 }}
                        animate={{
                          strokeDashoffset:
                            2 * Math.PI * 56 * (1 - result.confidence / 100),
                        }}
                        transition={{ duration: 1.5, ease: 'easeOut' }}
                        strokeLinecap="round"
                      />
                    </motion.svg>
                  </div>
                </div>
              </motion.div>

              {/* All-probabilities bar chart */}
              <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.55, duration: 0.6 }}
                className="mb-12 bg-slate-800/60 rounded-xl px-8 py-7 text-left"
              >
                <p className="text-slate-400 text-xs tracking-widest uppercase mb-6">
                  All Probabilities
                </p>

                <div className="flex flex-col gap-4">
                  {Object.entries(result.all_probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([emotion, prob], index) => {
                      const pct        = Math.round(prob * 100)
                      const isTop      = index === 0
                      const barColor   = isTop ? '#FFD8DF' : undefined

                      return (
                        <motion.div
                          key={emotion}
                          initial={{ opacity: 0, x: -12 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.6 + index * 0.07, duration: 0.4 }}
                          className="flex items-center gap-4"
                        >
                          {/* Label */}
                          <span
                            className={`w-24 text-sm font-medium capitalize shrink-0 ${
                              isTop ? 'text-white' : 'text-slate-400'
                            }`}
                          >
                            {emotion}
                          </span>

                          {/* Track */}
                          <div className="flex-1 h-2.5 bg-slate-700 rounded-full overflow-hidden">
                            <motion.div
                              className="h-full rounded-full"
                              style={{ backgroundColor: barColor ?? '#475569' }}
                              initial={{ width: '0%' }}
                              animate={{ width: `${pct}%` }}
                              transition={{
                                delay:    0.65 + index * 0.07,
                                duration: 0.9,
                                ease:     'easeOut',
                              }}
                            />
                          </div>

                          {/* Percentage */}
                          <span
                            className={`w-12 text-right text-sm font-semibold shrink-0 ${
                              isTop ? 'text-white' : 'text-slate-400'
                            }`}
                          >
                            {pct}%
                          </span>
                        </motion.div>
                      )
                    })}
                </div>
              </motion.div>

              {/* Reset */}
              <motion.button
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.7, duration: 0.6 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleReset}
                className="px-8 py-3 rounded-lg font-medium transition-all duration-200"
                style={{ backgroundColor: '#FFAAB8', color: 'white' }}
              >
                Analyze Another Image
              </motion.button>
            </motion.div>
          </section>
        )}
      </AnimatePresence>
    </main>
  )
}
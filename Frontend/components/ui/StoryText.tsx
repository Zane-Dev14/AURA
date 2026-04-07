'use client'
import { motion, AnimatePresence } from 'framer-motion'
import { useSceneStore } from '@/store/useSceneStore'

const STORY_TEXT: Record<string, { title: string; subtitle: string; description?: string }> = {
  calm: {
    title: 'A Peaceful Day',
    subtitle: 'KTU University Website - Normal Operations',
    description: 'Collect rings to explore the system'
  },
  system: {
    title: 'The Infrastructure',
    subtitle: 'Kubernetes Cluster Running Smoothly',
    description: '6 pods handling steady traffic'
  },
  traffic: {
    title: 'The Storm Arrives',
    subtitle: 'EXAM RESULTS RELEASED',
    description: '10,000 students accessing simultaneously'
  },
  failure: {
    title: 'System Overload',
    subtitle: 'HPA Scaling Too Slow',
    description: 'Pods failing under pressure - 4 seconds per pod'
  },
  emotional: {
    title: 'Critical Failure',
    subtitle: 'Students Unable to Access Results',
    description: 'System at breaking point'
  },
  qmix: {
    title: 'QMix Activation',
    subtitle: 'AI-Powered Predictive Scaling',
    description: 'Click nodes to initialize the system'
  },
  transform: {
    title: 'Transformation',
    subtitle: 'QMix Takes Control',
    description: 'Intelligent scaling in action'
  },
  recovery: {
    title: 'System Restored',
    subtitle: 'QMix Healing the Infrastructure',
    description: 'Pods recovering with optimal resource allocation'
  },
  comparison: {
    title: 'The Difference',
    subtitle: 'HPA vs QMix Performance',
    description: 'Toggle to compare approaches'
  }
}

export default function StoryText() {
  const { scene } = useSceneStore()
  const story = STORY_TEXT[scene]

  if (!story) return null

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={scene}
        className="story-text-overlay"
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 30 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      >
        <motion.div
          className="story-title"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
        >
          {story.title}
        </motion.div>
        
        <motion.div
          className="story-subtitle"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4, duration: 0.6 }}
        >
          {story.subtitle}
        </motion.div>
        
        {story.description && (
          <motion.div
            className="story-description"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6, duration: 0.6 }}
          >
            {story.description}
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  )
}

// Made with Bob

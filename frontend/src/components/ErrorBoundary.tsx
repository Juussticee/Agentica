import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { Button, ErrorCard } from './ui'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

/**
 * Last-resort recovery: a render crash (e.g. an unexpected payload shape)
 * must never leave a blank, unusable page. Shows the app's own error card
 * with a single action that returns to the dashboard.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Intentional console use: crash diagnostics belong in the browser
    // devtools so they are visible during debugging, not silently swallowed.
    console.error('Unhandled render error:', error, info.componentStack)
  }

  private recover = () => {
    // Point the hash back at the dashboard, then remount the app tree.
    if (window.location.hash !== '' && window.location.hash !== '#/' && window.location.hash !== '#') {
      window.location.hash = '#/'
    }
    this.setState({ error: null })
  }

  render() {
    if (this.state.error) {
      return (
        <div className="main-inner">
          <ErrorCard
            message="Something went wrong displaying this page."
            why={this.state.error.message}
            recovery="Return to the overview, then retry the action."
          />
          <Button type="button" onClick={this.recover}>Back to overview</Button>
        </div>
      )
    }
    return this.props.children
  }
}

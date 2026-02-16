import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Markdown from 'react-markdown';
import { useChat, useModelSelector } from '@nearstack-dev/react/ai';
import type { Note } from '../models/Note';
import {
  parseToolCalls,
  hasIncompleteToolCall,
  TOOL_DEFINITIONS,
  type ToolCall,
  type ParsedSegment,
} from '../lib/toolCalls';

interface AIPanelProps {
  notes: Note[];
  onCreateNote?: (args: { title: string; content: string; tags?: string[] }) => Promise<Note>;
  onUpdateNote?: (args: { id: string; title?: string; content?: string; tags?: string[] }) => Promise<Note | undefined>;
  onDeleteNote?: (args: { id: string }) => Promise<void>;
}

function buildSystemPrompt(notes: Note[]): string {
  const parts: string[] = [
    'You are a helpful AI assistant integrated into a personal notes app.',
    "You have access to the user's notes and can help find information, summarize content, suggest connections between notes, and answer questions.",
    'Be helpful and thorough. Reference specific notes by title when relevant.',
  ];

  if (notes.length > 0) {
    const notesSummary = notes
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, 10)
      .map(note => {
        const tags = note.tags.length > 0 ? ` [${note.tags.join(', ')}]` : '';
        return `- ID: ${note.id} | "${note.title || 'Untitled'}"${tags}: ${note.content.slice(0, 100)}`;
      })
      .join('\n');

    parts.push(`The user has ${notes.length} notes. Here are the most recent:\n${notesSummary}`);
  }

  parts.push(TOOL_DEFINITIONS);

  return parts.join('\n\n');
}

function ToolCallCard({
  toolCall,
  onApprove,
  onReject,
}: {
  toolCall: ToolCall;
  onApprove: (tc: ToolCall) => void;
  onReject: (tc: ToolCall) => void;
}) {
  const argsSummary = Object.entries(toolCall.args)
    .map(([key, value]) => {
      const display = typeof value === 'string' && value.length > 60
        ? value.slice(0, 60) + '...'
        : JSON.stringify(value);
      return `${key}: ${display}`;
    })
    .join(', ');

  const nameLabel: Record<string, string> = {
    create_note: 'Create Note',
    update_note: 'Update Note',
    delete_note: 'Delete Note',
  };

  return (
    <div className="my-2 border border-neutral-300 bg-neutral-50 p-3 text-sm">
      <div className="flex items-center justify-between">
        <span className="font-medium">{nameLabel[toolCall.name] || toolCall.name}</span>
        {toolCall.status === 'pending' && (
          <div className="flex gap-1.5">
            <button
              onClick={() => onApprove(toolCall)}
              className="bg-black px-2.5 py-1 text-xs text-white hover:bg-neutral-800"
            >
              Approve
            </button>
            <button
              onClick={() => onReject(toolCall)}
              className="border border-neutral-300 px-2.5 py-1 text-xs hover:bg-neutral-200"
            >
              Reject
            </button>
          </div>
        )}
        {toolCall.status === 'executed' && (
          <span className="text-xs text-neutral-500">Done</span>
        )}
        {toolCall.status === 'rejected' && (
          <span className="text-xs text-neutral-400">Rejected</span>
        )}
        {toolCall.status === 'error' && (
          <span className="text-xs text-red-600">Error</span>
        )}
      </div>
      <p className="mt-1 text-xs text-neutral-500">{argsSummary}</p>
      {toolCall.result && (
        <p className="mt-1 text-xs text-neutral-600">{toolCall.result}</p>
      )}
    </div>
  );
}

export function AIPanel({ notes, onCreateNote, onUpdateNote, onDeleteNote }: AIPanelProps) {
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const systemPrompt = useMemo(() => buildSystemPrompt(notes), [notes]);
  const { messages, send, isStreaming, error, clear } = useChat(undefined, { systemPrompt });
  const [toolCallStates, setToolCallStates] = useState<Map<string, ToolCall>>(new Map());
  const {
    choices,
    selectModel,
    downloadModel,
    isDownloading,
    downloadProgress,
    currentSelection,
    selectedModel,
  } = useModelSelector();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const getToolCallState = useCallback((tc: ToolCall): ToolCall => {
    return toolCallStates.get(tc.id) || tc;
  }, [toolCallStates]);

  const executeToolCall = useCallback(async (tc: ToolCall) => {
    setToolCallStates(prev => {
      const next = new Map(prev);
      next.set(tc.id, { ...tc, status: 'approved' });
      return next;
    });

    try {
      let result = '';
      switch (tc.name) {
        case 'create_note': {
          if (onCreateNote) {
            const note = await onCreateNote(tc.args as { title: string; content: string; tags?: string[] });
            result = `Created note "${note.title || 'Untitled'}"`;
          } else {
            result = 'Create note is not available';
          }
          break;
        }
        case 'update_note': {
          if (onUpdateNote) {
            const note = await onUpdateNote(tc.args as { id: string; title?: string; content?: string; tags?: string[] });
            result = note ? `Updated note "${note.title || 'Untitled'}"` : 'Note not found';
          } else {
            result = 'Update note is not available';
          }
          break;
        }
        case 'delete_note': {
          if (onDeleteNote) {
            await onDeleteNote(tc.args as { id: string });
            result = 'Note deleted';
          } else {
            result = 'Delete note is not available';
          }
          break;
        }
        default:
          result = `Unknown tool: ${tc.name}`;
      }

      setToolCallStates(prev => {
        const next = new Map(prev);
        next.set(tc.id, { ...tc, status: 'executed', result });
        return next;
      });
    } catch (err) {
      setToolCallStates(prev => {
        const next = new Map(prev);
        next.set(tc.id, { ...tc, status: 'error', result: err instanceof Error ? err.message : 'Failed' });
        return next;
      });
    }
  }, [onCreateNote, onUpdateNote, onDeleteNote]);

  const rejectToolCall = useCallback((tc: ToolCall) => {
    setToolCallStates(prev => {
      const next = new Map(prev);
      next.set(tc.id, { ...tc, status: 'rejected' });
      return next;
    });
  }, []);

  const renderAssistantMessage = useCallback((content: string) => {
    const parsed = parseToolCalls(content);
    const showIncomplete = isStreaming && hasIncompleteToolCall(content);

    return (
      <>
        {parsed.segments.map((segment: ParsedSegment, i: number) => {
          if (segment.type === 'text') {
            return (
              <div key={i} className="prose prose-sm prose-neutral max-w-none">
                <Markdown>{segment.content}</Markdown>
              </div>
            );
          }
          const tc = getToolCallState(segment.toolCall);
          return (
            <ToolCallCard
              key={segment.toolCallId}
              toolCall={tc}
              onApprove={executeToolCall}
              onReject={rejectToolCall}
            />
          );
        })}
        {showIncomplete && (
          <p className="mt-1 text-xs italic text-neutral-400">Preparing action...</p>
        )}
      </>
    );
  }, [isStreaming, getToolCallState, executeToolCall, rejectToolCall]);

  const needsSetup =
    !selectedModel ||
    (selectedModel.status?.state !== 'cached' && selectedModel.status?.state !== 'ready');

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-neutral-200 px-4 py-2">
        <h2 className="text-sm font-semibold">AI Assistant</h2>
        {messages.length > 0 && (
          <button onClick={() => { clear(); setToolCallStates(new Map()); }} className="text-xs text-neutral-400 hover:text-black">
            Clear
          </button>
        )}
      </div>

      {needsSetup ? (
        <div className="flex flex-1 flex-col items-center justify-center p-6 text-center">
          <p className="text-sm font-medium">Set up a local AI model</p>
          <p className="mt-1 text-xs text-neutral-500">
            Models run entirely on your device. No data leaves your browser.
          </p>

          <select
            className="mt-4 w-full border border-neutral-300 bg-white px-3 py-2 text-sm focus:border-black focus:outline-none"
            value={currentSelection ?? ''}
            onChange={e => {
              if (e.target.value) void selectModel(e.target.value);
            }}
            disabled={isDownloading}
          >
            <option value="">Select a model</option>
            {choices.map(choice => (
              <option key={choice.value} value={choice.value} disabled={choice.disabled}>
                {choice.group} &middot; {choice.label}
              </option>
            ))}
          </select>

          {selectedModel?.status?.state === 'available' && (
            <button
              onClick={() => void downloadModel(selectedModel.id)}
              className="mt-3 w-full bg-black px-3 py-2 text-sm text-white hover:bg-neutral-800"
            >
              Download model
            </button>
          )}

          {isDownloading && (
            <div className="mt-3 w-full">
              <div className="h-1 w-full overflow-hidden bg-neutral-200">
                <div
                  className="h-full bg-black transition-all"
                  style={{ width: `${Math.round(downloadProgress * 100)}%` }}
                />
              </div>
              <p className="mt-1 text-xs text-neutral-500">
                Downloading {Math.round(downloadProgress * 100)}%
              </p>
            </div>
          )}

          {selectedModel?.status?.state === 'error' && (
            <p className="mt-2 text-xs text-red-600">{selectedModel.status.message}</p>
          )}
        </div>
      ) : (
        <>
          <div className="flex-1 overflow-y-auto p-4">
            {messages.length === 0 && (
              <div className="flex h-full items-center justify-center">
                <div className="text-center text-neutral-400">
                  <p className="text-sm font-medium">Ask about your notes</p>
                  <div className="mt-3 space-y-1.5 text-xs">
                    <p className="text-neutral-500">&ldquo;Summarize my recent notes&rdquo;</p>
                    <p className="text-neutral-500">&ldquo;What topics come up most?&rdquo;</p>
                    <p className="text-neutral-500">&ldquo;Create a note about...&rdquo;</p>
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-3">
              {messages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={`text-sm ${
                    message.role === 'assistant'
                      ? ''
                      : 'ml-8 bg-neutral-100 px-3 py-2'
                  }`}
                >
                  {message.role === 'assistant' ? (
                    renderAssistantMessage(message.content)
                  ) : (
                    message.content
                  )}
                </div>
              ))}
              {error && <p className="text-xs text-red-600">{error}</p>}
            </div>
            <div ref={bottomRef} />
          </div>

          <form
            className="border-t border-neutral-200 p-3"
            onSubmit={async e => {
              e.preventDefault();
              const value = input.trim();
              if (!value) return;
              setInput('');
              await send(value, { maxTokens: 2048 });
            }}
          >
            <div className="flex gap-2">
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder="Ask anything..."
                className="flex-1 border border-neutral-300 bg-white px-3 py-2 text-sm placeholder:text-neutral-400 focus:border-black focus:outline-none"
              />
              <button
                type="submit"
                disabled={isStreaming}
                className="bg-black px-3 py-2 text-sm text-white hover:bg-neutral-800 disabled:opacity-50"
              >
                Send
              </button>
            </div>
          </form>
        </>
      )}
    </div>
  );
}

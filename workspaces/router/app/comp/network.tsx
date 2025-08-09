import React, { useState, useRef, useEffect } from 'react';

const InteractiveNetwork = () => {
  const [nodes, setNodes] = useState([
    { id: 1, x: 200, y: 150, label: 'Node 1', color: '#8B5CF6' },
    { id: 2, x: 400, y: 100, label: 'Node 2', color: '#06B6D4' },
    { id: 3, x: 300, y: 250, label: 'Node 3', color: '#10B981' },
    { id: 4, x: 500, y: 200, label: 'Node 4', color: '#F59E0B' },
  ]);

  const [connections, setConnections] = useState([
    { from: 1, to: 2 },
    { from: 2, to: 3 },
    { from: 3, to: 4 },
    { from: 1, to: 3 },
  ]);

  const [dragging, setDragging] = useState(null);
  const [connecting, setConnecting] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [newNodeLabel, setNewNodeLabel] = useState('');
  const svgRef = useRef(null);

  const colors = ['#8B5CF6', '#06B6D4', '#10B981', '#F59E0B', '#EC4899', '#EF4444'];

  const handleMouseDown = (e, nodeId) => {
    e.preventDefault();
    if (e.shiftKey) {
      if (connecting === null) {
        setConnecting(nodeId);
      } else if (connecting !== nodeId) {
        const newConnection = { from: connecting, to: nodeId };
        if (!connections.some(conn => 
          (conn.from === connecting && conn.to === nodeId) ||
          (conn.from === nodeId && conn.to === connecting)
        )) {
          setConnections([...connections, newConnection]);
        }
        setConnecting(null);
      }
    } else {
      setDragging(nodeId);
    }
  };

  const handleMouseMove = (e) => {
    if (dragging !== null) {
      const rect = svgRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      setNodes(nodes.map(node => 
        node.id === dragging ? { ...node, x, y } : node
      ));
    }
  };

  const handleMouseUp = () => {
    setDragging(null);
  };

  const addNode = () => {
    if (newNodeLabel.trim() === '') return;
    
    const newNode = {
      id: Math.max(...nodes.map(n => n.id)) + 1,
      x: Math.random() * 400 + 100,
      y: Math.random() * 200 + 100,
      label: newNodeLabel,
      color: colors[nodes.length % colors.length]
    };
    
    setNodes([...nodes, newNode]);
    setNewNodeLabel('');
  };

  const removeNode = (nodeId) => {
    setNodes(nodes.filter(node => node.id !== nodeId));
    setConnections(connections.filter(conn => conn.from !== nodeId && conn.to !== nodeId));
    if (connecting === nodeId) setConnecting(null);
    if (selectedNode === nodeId) {
      const remainingNodes = nodes.filter(node => node.id !== nodeId);
      setSelectedNode(remainingNodes.length > 0 ? remainingNodes[0].id : null);
    }
  };

  const clearConnections = () => {
    setConnections([]);
    setConnecting(null);
  };

  const moveSelectedNode = (direction) => {
    if (selectedNode === null) return;
    
    setNodes(nodes.map(node => {
      if (node.id === selectedNode) {
        let newX = node.x;
        let newY = node.y;
        
        switch (direction) {
          case 'ArrowUp':
            newY = Math.max(30, node.y - 10);
            break;
          case 'ArrowDown':
            newY = Math.min(370, node.y + 10);
            break;
          case 'ArrowLeft':
            newX = Math.max(30, node.x - 10);
            break;
          case 'ArrowRight':
            newX = Math.min(770, node.x + 10);
            break;
        }
        
        return { ...node, x: newX, y: newY };
      }
      return node;
    }));
  };

  useEffect(() => {
    const handleGlobalMouseUp = () => setDragging(null);
    
    const handleKeyDown = (e) => {
      if (e.code === 'Space' && nodes.length > 0) {
        e.preventDefault();
        if (selectedNode === null) {
          setSelectedNode(nodes[0].id);
        } else {
          const currentIndex = nodes.findIndex(node => node.id === selectedNode);
          const nextIndex = (currentIndex + 1) % nodes.length;
          setSelectedNode(nodes[nextIndex].id);
        }
      } else if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code) && selectedNode !== null) {
        e.preventDefault();
        moveSelectedNode(e.code);
      }
    };
    
    window.addEventListener('mouseup', handleGlobalMouseUp);
    window.addEventListener('keydown', handleKeyDown);
    
    return () => {
      window.removeEventListener('mouseup', handleGlobalMouseUp);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [nodes, selectedNode]);

  const getNodeById = (id) => nodes.find(node => node.id === id);

  return (
    <div className="w-full max-w-6xl mx-auto p-8 bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl shadow-2xl border border-slate-700">
      <div className="mb-8">
        <h2 className="text-4xl font-bold bg-gradient-to-r from-slate-200 to-slate-400 bg-clip-text text-transparent mb-6">
          Interactive Network
        </h2>
        
        <div className="flex flex-wrap gap-4 mb-6">
          <div className="flex items-center gap-3">
            <input
              type="text"
              value={newNodeLabel}
              onChange={(e) => setNewNodeLabel(e.target.value)}
              placeholder="Node name..."
              className="px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
              onKeyPress={(e) => e.key === 'Enter' && addNode()}
            />
            <button
              onClick={addNode}
              className="px-6 py-3 bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-lg hover:from-purple-700 hover:to-purple-800 transition-all duration-200 shadow-lg hover:shadow-purple-500/25"
            >
              Add Node
            </button>
          </div>
          
          <button
            onClick={clearConnections}
            className="px-6 py-3 bg-gradient-to-r from-slate-600 to-slate-700 text-white rounded-lg hover:from-slate-700 hover:to-slate-800 transition-all duration-200 shadow-lg"
          >
            Clear Connections
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-slate-400 mb-6 bg-slate-800/50 p-4 rounded-lg border border-slate-700">
          <div>
            <p className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
              Drag nodes to move
            </p>
            <p className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 bg-cyan-500 rounded-full"></span>
              Shift + Click to connect
            </p>
          </div>
          <div>
            <p className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 bg-emerald-500 rounded-full"></span>
              <kbd className="px-2 py-1 bg-slate-700 text-slate-300 text-xs rounded">Space</kbd> to select nodes
            </p>
            <p className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 bg-amber-500 rounded-full"></span>
              <kbd className="px-1 py-1 bg-slate-700 text-slate-300 text-xs rounded">↑↓←→</kbd> to move selected
            </p>
          </div>
        </div>
      </div>

      <div className="border border-slate-600 rounded-xl bg-slate-900/50 backdrop-blur-sm shadow-inner">
        <svg
          ref={svgRef}
          width="800"
          height="400"
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          className="cursor-crosshair focus:outline-none rounded-xl"
          tabIndex="0"
        >
          {/* グラデーション定義 */}
          <defs>
            <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#64748B" stopOpacity="0.8"/>
              <stop offset="100%" stopColor="#94A3B8" stopOpacity="0.4"/>
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
            <marker
              id="arrowhead"
              markerWidth="12"
              markerHeight="8"
              refX="10"
              refY="4"
              orient="auto"
            >
              <polygon
                points="0 0, 12 4, 0 8"
                fill="#94A3B8"
                opacity="0.8"
              />
            </marker>
          </defs>

          {/* 背景グリッド */}
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#334155" strokeWidth="0.5" opacity="0.3"/>
          </pattern>
          <rect width="800" height="400" fill="url(#grid)" />

          {/* 接続線を描画 */}
          {connections.map((conn, index) => {
            const fromNode = getNodeById(conn.from);
            const toNode = getNodeById(conn.to);
            if (!fromNode || !toNode) return null;

            return (
              <line
                key={index}
                x1={fromNode.x}
                y1={fromNode.y}
                x2={toNode.x}
                y2={toNode.y}
                stroke="url(#connectionGradient)"
                strokeWidth="2"
                markerEnd="url(#arrowhead)"
                className="drop-shadow-sm"
              />
            );
          })}

          {/* 接続中のライン */}
          {connecting !== null && (
            <line
              x1={getNodeById(connecting).x}
              y1={getNodeById(connecting).y}
              x2={getNodeById(connecting).x}
              y2={getNodeById(connecting).y}
              stroke="#8B5CF6"
              strokeWidth="3"
              strokeDasharray="8,4"
              opacity="0.8"
            />
          )}

          {/* ノードを描画 */}
          {nodes.map((node) => (
            <g key={`node-${node.id}`}>
              {/* 選択されたノードの光るアニメーション */}
              {selectedNode === node.id && (
                <g key={`selected-${node.id}`}>
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r="40"
                    fill="none"
                    stroke="#8B5CF6"
                    strokeWidth="2"
                    opacity="0.6"
                    filter="url(#glow)"
                  >
                    <animate
                      attributeName="r"
                      values="35;45;35"
                      dur="2s"
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      values="0.6;0.2;0.6"
                      dur="2s"
                      repeatCount="indefinite"
                    />
                  </circle>
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r="32"
                    fill="none"
                    stroke="#A855F7"
                    strokeWidth="1"
                    opacity="0.4"
                  >
                    <animate
                      attributeName="opacity"
                      values="0.4;0.8;0.4"
                      dur="1.5s"
                      repeatCount="indefinite"
                    />
                  </circle>
                </g>
              )}
              
              {/* メインノード */}
              <circle
                cx={node.x}
                cy={node.y}
                r="30"
                fill={node.color}
                stroke={connecting === node.id ? "#A855F7" : selectedNode === node.id ? "#8B5CF6" : "#475569"}
                strokeWidth={connecting === node.id || selectedNode === node.id ? "3" : "2"}
                className="cursor-pointer hover:brightness-110 transition-all duration-200 drop-shadow-lg"
                onMouseDown={(e) => handleMouseDown(e, node.id)}
                onDoubleClick={() => removeNode(node.id)}
                filter={selectedNode === node.id ? "url(#glow)" : "none"}
              />
              
              {/* ノードラベル */}
              <text
                x={node.x}
                y={node.y + 5}
                textAnchor="middle"
                className="text-white text-sm font-semibold pointer-events-none select-none drop-shadow-sm"
              >
                {node.label}
              </text>
              
              {/* ノードID表示 */}
              <text
                x={node.x}
                y={node.y - 42}
                textAnchor="middle"
                className={`text-xs pointer-events-none select-none font-medium transition-colors duration-200 ${
                  selectedNode === node.id ? 'text-purple-400 drop-shadow-sm' : 'text-slate-500'
                }`}
              >
                #{node.id} {selectedNode === node.id ? '◉' : ''}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className="mt-6 p-6 bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700">
        <h3 className="text-xl font-semibold text-slate-200 mb-4 flex items-center gap-2">
          <div className="w-3 h-3 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full"></div>
          Network Status
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
          <div className="text-center p-4 bg-slate-700/30 rounded-lg border border-slate-600">
            <div className="text-2xl font-bold text-purple-400">{nodes.length}</div>
            <div className="text-slate-400">Nodes</div>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg border border-slate-600">
            <div className="text-2xl font-bold text-cyan-400">{connections.length}</div>
            <div className="text-slate-400">Connections</div>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg border border-slate-600">
            <div className="text-2xl font-bold text-emerald-400">
              {selectedNode !== null ? `#${selectedNode}` : '—'}
            </div>
            <div className="text-slate-400">Selected</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InteractiveNetwork;
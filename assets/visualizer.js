class ArenaVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.agents = [];
        this.animationId = null;
        this.centerX = 0;
        this.centerY = 0;
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth || 400;
        this.canvas.height = container.clientHeight || 400;
        this.centerX = this.canvas.width / 2;
        this.centerY = this.canvas.height / 2;
    }

    setAgents(agentData) {
        this.agents = agentData.map((data, index) => ({
            id: data.id || `agent_${index}`,
            name: data.name || `Agent ${index + 1}`,
            dominance: data.dominance || 50,
            targetDominance: data.dominance || 50,
            color: data.color || this.getDefaultColor(index),
            x: this.centerX + (Math.random() - 0.5) * 100,
            y: this.centerY + (Math.random() - 0.5) * 100,
            vx: 0,
            vy: 0,
            targetX: this.centerX,
            targetY: this.centerY,
            pulsePhase: Math.random() * Math.PI * 2,
            glowIntensity: 0,
        }));
        this.calculatePositions();
    }

    getDefaultColor(index) {
        const colors = [
            { primary: '#ef4444', secondary: '#fca5a5', glow: 'rgba(239, 68, 68, 0.6)' },
            { primary: '#3b82f6', secondary: '#93c5fd', glow: 'rgba(59, 130, 246, 0.6)' },
            { primary: '#10b981', secondary: '#6ee7b7', glow: 'rgba(16, 185, 129, 0.6)' },
            { primary: '#f59e0b', secondary: '#fcd34d', glow: 'rgba(245, 158, 11, 0.6)' },
        ];
        return colors[index % colors.length];
    }

    updateDominance(agentId, newDominance) {
        const agent = this.agents.find(a => a.id === agentId || a.name === agentId);
        if (agent) {
            agent.targetDominance = Math.max(0, Math.min(100, newDominance));
            agent.glowIntensity = 1;
        }
        this.calculatePositions();
    }

    updateAllDominance(dominanceMap) {
        Object.entries(dominanceMap).forEach(([name, dominance]) => {
            const agent = this.agents.find(a => a.name === name || a.id === name);
            if (agent) {
                const oldDominance = agent.targetDominance;
                agent.targetDominance = Math.max(0, Math.min(100, dominance));
                if (Math.abs(oldDominance - dominance) > 2) {
                    agent.glowIntensity = 1;
                }
            }
        });
        this.calculatePositions();
    }

    calculatePositions() {
        if (this.agents.length === 0) return;
        
        const totalDominance = this.agents.reduce((sum, a) => sum + a.targetDominance, 0);
        const maxRadius = Math.min(this.canvas.width, this.canvas.height) * 0.35;
        
        this.agents.sort((a, b) => b.targetDominance - a.targetDominance);
        
        this.agents.forEach((agent, index) => {
            const dominanceRatio = agent.targetDominance / totalDominance;
            const distanceFromCenter = maxRadius * (1 - dominanceRatio * 1.5);
            const angle = (index / this.agents.length) * Math.PI * 2 - Math.PI / 2;
            
            if (index === 0 && agent.targetDominance > totalDominance * 0.6) {
                agent.targetX = this.centerX;
                agent.targetY = this.centerY;
            } else {
                agent.targetX = this.centerX + Math.cos(angle) * Math.max(30, distanceFromCenter);
                agent.targetY = this.centerY + Math.sin(angle) * Math.max(30, distanceFromCenter);
            }
        });
    }

    getRadius(dominance) {
        const minRadius = 30;
        const maxRadius = Math.min(this.canvas.width, this.canvas.height) * 0.2;
        return minRadius + (dominance / 100) * (maxRadius - minRadius);
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawBackground();
        
        const dt = 0.016;
        const springStrength = 0.08;
        const damping = 0.85;
        
        this.agents.forEach(agent => {
            agent.dominance += (agent.targetDominance - agent.dominance) * 0.1;
            
            const dx = agent.targetX - agent.x;
            const dy = agent.targetY - agent.y;
            agent.vx += dx * springStrength;
            agent.vy += dy * springStrength;
            agent.vx *= damping;
            agent.vy *= damping;
            agent.x += agent.vx;
            agent.y += agent.vy;
            
            agent.pulsePhase += 0.05;
            agent.glowIntensity *= 0.95;
        });
        
        this.applyRepulsion();
        this.drawConnections();
        this.agents.forEach(agent => this.drawAgent(agent));
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }

    applyRepulsion() {
        const repulsionStrength = 0.5;
        
        for (let i = 0; i < this.agents.length; i++) {
            for (let j = i + 1; j < this.agents.length; j++) {
                const a1 = this.agents[i];
                const a2 = this.agents[j];
                
                const dx = a2.x - a1.x;
                const dy = a2.y - a1.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                const r1 = this.getRadius(a1.dominance);
                const r2 = this.getRadius(a2.dominance);
                const minDist = r1 + r2 + 20;
                
                if (dist < minDist && dist > 0) {
                    const force = (minDist - dist) * repulsionStrength / dist;
                    const fx = dx * force;
                    const fy = dy * force;
                    
                    a1.vx -= fx;
                    a1.vy -= fy;
                    a2.vx += fx;
                    a2.vy += fy;
                }
            }
        }
    }

    drawBackground() {
        const gradient = this.ctx.createRadialGradient(
            this.centerX, this.centerY, 0,
            this.centerX, this.centerY, Math.max(this.canvas.width, this.canvas.height) * 0.7
        );
        gradient.addColorStop(0, '#1e1b4b');
        gradient.addColorStop(0.5, '#0f172a');
        gradient.addColorStop(1, '#020617');
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.strokeStyle = 'rgba(99, 102, 241, 0.1)';
        this.ctx.lineWidth = 1;
        const rings = 5;
        const maxRing = Math.min(this.canvas.width, this.canvas.height) * 0.4;
        for (let i = 1; i <= rings; i++) {
            this.ctx.beginPath();
            this.ctx.arc(this.centerX, this.centerY, (maxRing / rings) * i, 0, Math.PI * 2);
            this.ctx.stroke();
        }
    }

    drawConnections() {
        if (this.agents.length < 2) return;
        
        for (let i = 0; i < this.agents.length; i++) {
            for (let j = i + 1; j < this.agents.length; j++) {
                const a1 = this.agents[i];
                const a2 = this.agents[j];
                
                const gradient = this.ctx.createLinearGradient(a1.x, a1.y, a2.x, a2.y);
                gradient.addColorStop(0, `${a1.color.primary}40`);
                gradient.addColorStop(1, `${a2.color.primary}40`);
                
                this.ctx.beginPath();
                this.ctx.strokeStyle = gradient;
                this.ctx.lineWidth = 2;
                this.ctx.moveTo(a1.x, a1.y);
                this.ctx.lineTo(a2.x, a2.y);
                this.ctx.stroke();
            }
        }
    }

    drawAgent(agent) {
        const radius = this.getRadius(agent.dominance);
        const pulseOffset = Math.sin(agent.pulsePhase) * 3;
        const currentRadius = radius + pulseOffset;
        
        if (agent.glowIntensity > 0.1) {
            const glowRadius = currentRadius + 20 * agent.glowIntensity;
            const glow = this.ctx.createRadialGradient(
                agent.x, agent.y, currentRadius,
                agent.x, agent.y, glowRadius
            );
            glow.addColorStop(0, agent.color.glow);
            glow.addColorStop(1, 'transparent');
            this.ctx.fillStyle = glow;
            this.ctx.beginPath();
            this.ctx.arc(agent.x, agent.y, glowRadius, 0, Math.PI * 2);
            this.ctx.fill();
        }
        
        const gradient = this.ctx.createRadialGradient(
            agent.x - currentRadius * 0.3, agent.y - currentRadius * 0.3, 0,
            agent.x, agent.y, currentRadius
        );
        gradient.addColorStop(0, agent.color.secondary);
        gradient.addColorStop(0.7, agent.color.primary);
        gradient.addColorStop(1, this.darkenColor(agent.color.primary, 0.3));
        
        this.ctx.beginPath();
        this.ctx.arc(agent.x, agent.y, currentRadius, 0, Math.PI * 2);
        this.ctx.fillStyle = gradient;
        this.ctx.fill();
        
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = `bold ${Math.max(12, currentRadius * 0.3)}px "Microsoft YaHei", sans-serif`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        
        this.ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
        this.ctx.shadowBlur = 4;
        this.ctx.fillText(agent.name, agent.x, agent.y - 8);
        
        this.ctx.font = `${Math.max(10, currentRadius * 0.25)}px "Microsoft YaHei", sans-serif`;
        this.ctx.fillText(`${Math.round(agent.dominance)}`, agent.x, agent.y + 12);
        this.ctx.shadowBlur = 0;
    }

    darkenColor(hex, factor) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgb(${Math.floor(r * (1 - factor))}, ${Math.floor(g * (1 - factor))}, ${Math.floor(b * (1 - factor))})`;
    }

    start() {
        if (!this.animationId) {
            this.animate();
        }
    }

    stop() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
}

let visualizer = null;

function initVisualizer() {
    const canvas = document.getElementById('arena-canvas');
    if (canvas && !visualizer) {
        visualizer = new ArenaVisualizer('arena-canvas');
        visualizer.start();
    }
    return visualizer;
}

function updateVisualizerAgents(agentData) {
    const vis = initVisualizer();
    if (vis) {
        vis.setAgents(agentData);
    }
}

function updateVisualizerDominance(dominanceMap) {
    if (visualizer) {
        visualizer.updateAllDominance(dominanceMap);
    }
}

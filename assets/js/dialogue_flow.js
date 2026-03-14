// dialogue flow and turn scheduler

let utteranceBlackboard=[];
let utteranceSeq=1;

function resetUtteranceBlackboard(list){
    utteranceBlackboard=[];
    pendingUtterances=utteranceBlackboard;
    for(const u of (list||[])){
        const item={...u,_bb_seq:utteranceSeq++};
        utteranceBlackboard.push(item);
        if(item.tts_url){
            const npc=chars.find(c=>c.n===item.npc_id||c.a===item.npc_id);
            const speaker=npc?npc.n:item.npc_id;
            seedNpcSpeechCache(item.text,detectEmotionLabel(item.text),speaker,item.tts_url);
        }
    }
    if(utteranceBlackboard.length>0){
        const first=utteranceBlackboard[0];
        const n1=chars.find(c=>c.n===first.npc_id||c.a===first.npc_id);
        const s1=n1?n1.n:first.npc_id;
        prefetchNpcSpeech(first.text,detectEmotionLabel(first.text),s1);
        if(utteranceBlackboard.length>1){
            const second=utteranceBlackboard[1];
            const n2=chars.find(c=>c.n===second.npc_id||c.a===second.npc_id);
            const s2=n2?n2.n:second.npc_id;
            prefetchNpcSpeech(second.text,detectEmotionLabel(second.text),s2);
        }
    }
}

async function start(){
chars=mems.map(m=>ensureMemberVisuals({...m}));
show('p3');
await unlockNpcAudio();
$('cl').innerHTML=chars.map(c=>buildHeadCard(c)).join('');
renderConversationState('idle');
runNonverbalLoop();
updScr(50,50);
setCoachHint('');

// 处理面试问题显示
const interviewQuestionBox = $('interviewQuestionBox');
const interviewQuestionContent = $('interviewQuestionContent');
const sceneDescription = $('sceneDescriptionEdit')?.value || $('sceneDescriptionText')?.innerText || '';

if(scene === '群面竞争场'){
    interviewQuestionContent.textContent = sceneDescription;
    interviewQuestionContent.classList.remove('collapsed');
    interviewQuestionBox.style.display = 'block';
    document.querySelector('.interview-question-toggle').classList.remove('collapsed');
    document.getElementById('toggleArrow').textContent = '▼';
} else {
    interviewQuestionBox.style.display = 'none';
}

try{const r=await fetch('/api/session/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario_id:selectedScenarioId,scene_name:scene,characters:chars,scene_description:sceneDescription,user_info:(window.userInfo||null),pressure_tags:selectedPressureTags,pressure_value:pressureValue,drinking_capacity:drinkingCapacity})});
const d=await r.json();if(!d.success){alert(d.error);return}
sid=d.data.session_id;
if(d.data.is_unified_agent){
    resetUtteranceBlackboard(d.data.utterances||[]);
    shouldAwaitUser=d.data.should_await_user!==false;
    if(utteranceBlackboard.length>0){
        displayUtterances();
    }else if(shouldAwaitUser){
        shouldAwaitUser=true;
        $('interruptBtn').style.display='none';
    }
}else{
    if(d.data.opening_utterances&&d.data.opening_utterances.length>0){
        resetUtteranceBlackboard(d.data.opening_utterances||[]);
        shouldAwaitUser=true;
        displayUtterances();
    }else if(d.data.opening){
        addBot(d.data.opening,d.data.opening_speaker||null,detectEmotion(d.data.opening));
    }
}
}catch(e){alert(e)}
}

function toggleInterviewQuestionBox(){
    const content = $('interviewQuestionContent');
    const toggle = document.querySelector('.interview-question-toggle');
    const arrow = document.getElementById('toggleArrow');
    
    if(content.classList.contains('collapsed')){
        content.classList.remove('collapsed');
        toggle.classList.remove('collapsed');
        arrow.textContent = '▼';
    } else {
        content.classList.add('collapsed');
        toggle.classList.add('collapsed');
        arrow.textContent = '▲';
    }
}
function setCoachHint(text){
    const box=$('cb');
    const content=$('ct2');
    if(!box||!content)return;
    const value=String(text||'').trim();
    content.textContent=value;
    box.style.display=value?'flex':'none';
}
async function send(){
const t=$('ci2').value.trim();if(!t||!sid)return;$('ci2').value='';stopNpcVoice();renderConversationState('user_speaking');addUser(t);
await unlockNpcAudio();
const multimodal={emotion:emotionData,voice_features:lastVoiceFeatures||null,voice_text:lastVoiceText||''};
console.log('[Send] 消息:', t);console.log('[Send] 情感数据:', multimodal);
try{const r=await fetch('/api/chat/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,message:t,multimodal:multimodal})});
const d=await r.json();console.log('[Chat] 响应:', JSON.stringify(d, null, 2));if(d.success){
    if(d.data.utterances){
        resetUtteranceBlackboard(d.data.utterances||[]);
        shouldAwaitUser=d.data.should_await_user!==false;
        if(utteranceBlackboard.length>0){
            displayUtterances();
        }else if(shouldAwaitUser){
            shouldAwaitUser=true;
            $('interruptBtn').style.display='none';
        }
        let judge=d.data.judgment||'';
        if(judge&&d.data.npc_feedback_quality&&d.data.npc_feedback_quality.label){judge+=`（质量：${d.data.npc_feedback_quality.label}）`}
        setCoachHint(judge);
        if(d.data.new_dominance)updScr(d.data.new_dominance.user,d.data.new_dominance.ai);
        if(d.data.scores)updateMetrics(d.data.scores);
        if(d.data.game_over)scheduleEndSession(2000);
    }else{
        if(d.data.ai_text)addBot(d.data.ai_text,d.data.speaker,detectEmotion(d.data.ai_text));
        let judge=d.data.judgment||'';
        if(judge&&d.data.npc_feedback_quality&&d.data.npc_feedback_quality.label){judge+=`（质量：${d.data.npc_feedback_quality.label}）`}
        setCoachHint(judge);
        updScr(d.data.new_dominance.user,d.data.new_dominance.ai);
        updateMetrics(d.data.scores);
        if(d.data.game_over)scheduleEndSession(2000)
    }
    if(d.data.multimodal_analysis&&d.data.multimodal_analysis.emotion_state){
        const em=d.data.multimodal_analysis.emotion_state.primary_emotion||'neutral';
        lastVoiceEmotion=em;
        setVoiceEmotion(em);
    }
}}catch(e){console.log('[Chat] 错误:', e)}
lastVoiceFeatures=null;
lastVoiceText='';

}
function addUser(t){hist.push({role:'user',content:t});const c=$('mc2');c.innerHTML+=`<div class="msg u"><div class="mco">${t}</div></div>`;c.scrollTop=c.scrollHeight}

function addBot(t,sp,emo){return addBotStreaming(t,sp,emo)}

function addBotStreaming(t,sp,emo){
    hist.push({role:'assistant',content:t});
    const c=$('mc2');
    const msgId='msg-'+Date.now();
    c.innerHTML+=`<div class="msg b" id="${msgId}">${sp?`<div class="ms">${sp}</div>`:''}${emo?`<span class="msg-emo">${emo}</span>`:''}<div class="mco"></div></div>`;
    c.scrollTop=c.scrollHeight;
    
    const speaker=sp||chars[0]?.n||'';
    if(speaker){
        lastSpeaker=speaker;
        renderConversationState('npc_speaking',speaker);
        const card=document.querySelector(`.ci[data-n="${speaker}"] .ca`);
        if(card){card.style.transform='scale(1.12)';setTimeout(()=>card.style.transform='scale(1)',220)}
    }else{
        renderConversationState('idle');
    }
    
    speakNPC(t, detectEmotionLabel(t), speaker);
    const mco=document.querySelector(`#${msgId} .mco`);
    let idx=0;
    return new Promise((resolve)=>{
        const typeChar=()=>{
            if(idx<t.length){
                mco.textContent+=t.charAt(idx);
                idx++;
                c.scrollTop=c.scrollHeight;
                setTimeout(typeChar,10);
            }else{
                if(speaker){
                    setTimeout(()=>renderConversationState('after_npc',speaker),700);
                }
                resolve();
            }
        };
        typeChar();
    });
}

async function displayUtterances(){
    if(utteranceBlackboard.length===0){
        if(shouldAwaitUser){
            isNPCSpeaking=false;
            $('interruptBtn').style.display='none';
        }else{
            await continueNPC();
        }
        return;
    }
    
    isNPCSpeaking=true;
    $('interruptBtn').style.display='inline-flex';
    
    const utterance=utteranceBlackboard.shift();
    console.log('[displayUtterances] utterance:', utterance);
    console.log('[displayUtterances] chars:', chars);
    const npc=chars.find(c=>c.n===utterance.npc_id||c.a===utterance.npc_id);
    const speakerName=npc?npc.n:utterance.npc_id;
    console.log('[displayUtterances] speakerName:', speakerName);

    if(utteranceBlackboard.length>0){
        const next=utteranceBlackboard[0];
        const nextNpc=chars.find(c=>c.n===next.npc_id||c.a===next.npc_id);
        const nextSpeaker=nextNpc?nextNpc.n:next.npc_id;
        prefetchNpcSpeech(next.text,detectEmotionLabel(next.text),nextSpeaker);
    }
    await waitForAudioTurn(speakerName, Number(utterance.delay_ms)||700);
    
    await addBotStreaming(utterance.text,speakerName,detectEmotion(utterance.text));
    
    const delay=Math.max(300,Math.min(900,Number(utterance.delay_ms)||700));
    setTimeout(displayUtterances,delay);
}

async function interrupt(){
    if(!isNPCSpeaking)return;
    
    stopNpcVoice();
    resetUtteranceBlackboard([]);
    isNPCSpeaking=false;
    $('interruptBtn').style.display='none';
    
    try{
        const r=await fetch('/api/chat/interrupt',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({session_id:sid})
        });
        const d=await r.json();
        if(d.success){
            resetUtteranceBlackboard(d.data.utterances||[]);
            shouldAwaitUser=d.data.should_await_user!==false;
        }
    }catch(e){
        console.log('[Interrupt] 错误:',e);
    }
}

async function continueNPC(){
    try{
        const r=await fetch('/api/chat/continue',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({session_id:sid})
        });
        const d=await r.json();
        if(d.success){
            resetUtteranceBlackboard(d.data.utterances||[]);
            shouldAwaitUser=d.data.should_await_user!==false;
            if(utteranceBlackboard.length>0){
                displayUtterances();
            }
        }
    }catch(e){
        console.log('[Continue] 错误:',e);
    }
}


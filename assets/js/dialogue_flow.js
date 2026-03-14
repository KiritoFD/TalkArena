// dialogue flow and turn scheduler

let utteranceBlackboard=[];
let utteranceSeq=1;
let llmBusyCount=0;
let llmBusyTimer=null;
let llmBusyTick=0;

function _kickoffAudioPipeline(list){
    if(!Array.isArray(list)||!list.length)return;
    list.slice(0,8).forEach((u,idx)=>{
        setTimeout(()=>{
            try{
                const npc=chars.find(c=>c.n===u.npc_id||c.a===u.npc_id);
                const speaker=npc?npc.n:u.npc_id;
                const emo=String(u.emotion||'neutral');
                if(typeof prepareNpcSpeech==='function'){
                    u._preparedAudioPromise=prepareNpcSpeech(u.text,emo,speaker,u.tts_url||'');
                }else{
                    prefetchNpcSpeech(u.text,emo,speaker);
                }
            }catch(e){}
        },idx*35);
    });
}

function buildChatHistoryPayload(){
    if(!Array.isArray(hist)||!hist.length)return [];
    return hist
        .filter(x=>x&&typeof x==='object'&&String(x.content||'').trim())
        .map(x=>({
            role:String(x.role||'assistant'),
            speaker:String(x.speaker||'').trim(),
            content:String(x.content||'').trim()
        }));
}

function _ensureLlmBusyUi(){
    if(document.getElementById('llmBusyStyle'))return;
    const style=document.createElement('style');
    style.id='llmBusyStyle';
    style.textContent=`
#llmBusyBar{position:fixed;left:0;right:0;top:0;height:3px;z-index:5000;display:none;background:rgba(37,99,235,.15);overflow:hidden}
#llmBusyBar::after{content:'';display:block;width:35%;height:100%;background:linear-gradient(90deg,#2563eb,#60a5fa);animation:llmBusySlide 1s linear infinite}
@keyframes llmBusySlide{0%{transform:translateX(-120%)}100%{transform:translateX(320%)}}
`;
    document.head.appendChild(style);
    const bar=document.createElement('div');
    bar.id='llmBusyBar';
    document.body.appendChild(bar);
}

function _setLlmBusy(active,label='时间暂停，众人思考中'){
    _ensureLlmBusyUi();
    llmBusyCount=Math.max(0,llmBusyCount+(active?1:-1));
    const busy=llmBusyCount>0;
    const bar=document.getElementById('llmBusyBar');
    if(bar)bar.style.display=busy?'block':'none';

    const input=$('ci2');
    const sendBtn=document.querySelector('.sb');
    const rescueBtn=document.querySelector('.rescue-fab');
    if(sendBtn)sendBtn.disabled=busy;
    if(rescueBtn)rescueBtn.disabled=busy;
    if(input)input.readOnly=busy;

    if(!busy){
        if(llmBusyTimer){clearInterval(llmBusyTimer);llmBusyTimer=null;}
        if(input)input.placeholder='输入消息...';
        if(sendBtn)sendBtn.textContent='发送';
        return;
    }
    if(llmBusyTimer)return;
    llmBusyTick=0;
    llmBusyTimer=setInterval(()=>{
        llmBusyTick=(llmBusyTick+1)%4;
        const dots='.'.repeat(llmBusyTick);
        if(input)input.placeholder=`${label}${dots}`;
        if(sendBtn)sendBtn.textContent=`请稍等${dots}`;
    },360);
}
window.__setLlmBusy=_setLlmBusy;

function resetUtteranceBlackboard(list){
    utteranceBlackboard=[];
    pendingUtterances=utteranceBlackboard;
    for(const u of (list||[])){
        const item={...u,_bb_seq:utteranceSeq++};
        utteranceBlackboard.push(item);
        if(item.tts_url){
            const npc=chars.find(c=>c.n===item.npc_id||c.a===item.npc_id);
            const speaker=npc?npc.n:item.npc_id;
            seedNpcSpeechCache(item.text,String(item.emotion||'neutral'),speaker,item.tts_url);
        }
    }
    _kickoffAudioPipeline(utteranceBlackboard);
}

async function start(){
const startT0=performance.now();
console.log('[StartFlow] click_start');
chars=mems.map(m=>ensureMemberVisuals({...m}));
show('p3');
console.log(`[StartFlow] show_p3 cost_ms=${Math.round(performance.now()-startT0)}`);
await unlockNpcAudio();
console.log(`[StartFlow] unlock_audio_done cost_ms=${Math.round(performance.now()-startT0)}`);
if(typeof toggleC==='function'&&!isC){
    try{
        toggleC(true).then(()=>{
            console.log(`[StartFlow] camera_auto_open_async_done isC=${isC?1:0} total_ms=${Math.round(performance.now()-startT0)}`);
        }).catch(()=>{});
    }catch(e){}
}
console.log(`[StartFlow] camera_auto_open done isC=${isC?1:0} cost_ms=${Math.round(performance.now()-startT0)}`);
$('cl').innerHTML=chars.map(c=>buildHeadCard(c)).join('');
renderConversationState('idle');
runNonverbalLoop();
updScr(50,50);
updateMetrics(null);
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

try{
const apiT0=performance.now();
const r=await fetch('/api/session/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario_id:selectedScenarioId,scene_name:scene,characters:chars,scene_description:sceneDescription,user_info:(window.userInfo||null),pressure_tags:selectedPressureTags,pressure_value:pressureValue,drinking_capacity:drinkingCapacity})});
const d=await r.json();
console.log(`[StartFlow] api_session_start_done cost_ms=${Math.round(performance.now()-apiT0)} total_ms=${Math.round(performance.now()-startT0)} meta=`,d.meta||{});
if(!d.success){alert(d.error);return}
sid=d.data.session_id;
if(d.data.is_unified_agent){
    resetUtteranceBlackboard(d.data.utterances||[]);
    const total=(d.data.utterances||[]).length;
    const withUrl=(d.data.utterances||[]).filter(u=>String(u?.tts_url||'').trim()).length;
    console.log(`[StartFlow] unified_openings utterances=${total} with_tts_url=${withUrl}`);
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
_setLlmBusy(true,'时间暂停，众人思考中');
try{const r=await fetch('/api/chat/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,message:t,multimodal:multimodal,chat_history:buildChatHistoryPayload(),scenario_id:selectedScenarioId,scene_name:scene,scene_description:($('sceneDescriptionEdit')?.value||$('sceneDescriptionText')?.innerText||''),characters:chars})});
const d=await r.json();console.log('[Chat] 响应:', JSON.stringify(d, null, 2));if(d.success){
    window.__lastChatResponseAt=performance.now();
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
finally{_setLlmBusy(false)}
lastVoiceFeatures=null;
lastVoiceText='';

}
function addUser(t){hist.push({role:'user',speaker:'用户',content:t});const c=$('mc2');c.innerHTML+=`<div class="msg u"><div class="mco">${t}</div></div>`;c.scrollTop=c.scrollHeight}

function addBot(t,sp,emo,opts){return addBotStreaming(t,sp,emo,opts)}

function addBotStreaming(t,sp,emo,opts={}){
    hist.push({role:'assistant',speaker:sp||'',content:t});
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
    
    const speechEmotion=String(opts.emotion||'').trim()||detectEmotionLabel(t);
    if(typeof enqueueNPCSpeech==='function'){
        enqueueNPCSpeech(t, speechEmotion, speaker, {
            preferredUrl:opts.ttsUrl||'',
            preparedPromise:opts.preparedPromise||null
        });
    }else{
        speakNPC(t, speechEmotion, speaker, {
            preferredUrl:opts.ttsUrl||'',
            preparedPromise:opts.preparedPromise||null
        });
    }
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
    
    const lineT0=performance.now();
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
        if(!next._preparedAudioPromise&&typeof prepareNpcSpeech==='function'){
            next._preparedAudioPromise=prepareNpcSpeech(
                next.text,
                String(next.emotion||'neutral'),
                nextSpeaker,
                next.tts_url||''
            );
        }else{
            prefetchNpcSpeech(next.text,String(next.emotion||'neutral'),nextSpeaker);
        }
    }
    await waitForAudioTurn(speakerName, Number(utterance.delay_ms)||700);
    console.log(`[displayUtterances] wait_for_audio_done speaker=${speakerName} cost_ms=${Math.round(performance.now()-lineT0)} has_tts_url=${utterance?.tts_url?1:0}`);
    
    await addBotStreaming(
        utterance.text,
        speakerName,
        detectEmotion(utterance.text),
        {
            ttsUrl:utterance.tts_url||'',
            emotion:utterance.emotion||'neutral',
            preparedPromise:utterance._preparedAudioPromise||null
        }
    );
    console.log(`[displayUtterances] addBotStreaming_done speaker=${speakerName} cost_ms=${Math.round(performance.now()-lineT0)}`);
    
    const delay=80;
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
            body:JSON.stringify({
                session_id:sid,
                chat_history:buildChatHistoryPayload(),
                scenario_id:selectedScenarioId,
                scene_name:scene,
                scene_description:($('sceneDescriptionEdit')?.value||$('sceneDescriptionText')?.innerText||''),
                characters:chars
            })
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
            body:JSON.stringify({
                session_id:sid,
                chat_history:buildChatHistoryPayload(),
                scenario_id:selectedScenarioId,
                scene_name:scene,
                scene_description:($('sceneDescriptionEdit')?.value||$('sceneDescriptionText')?.innerText||''),
                characters:chars
            })
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


function step1_params_geom()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'step1_params_geom.log']);
  try
    m=mphload([base 'ARC_TA_loss.mph']); fprintf('LOADED %s\n', datestr(now,31));
    fprintf('OLD: Np=%s Nt=%s Ip=%s T=%s\n', char(m.param.get('Np')), char(m.param('par6').get('Nt')), ...
            char(m.param('par6').get('Ip')), char(m.param('par6').get('T')));
    % --- 第1步: 只改参数 ---
    m.param.set('Np','12');               % 饼数 Nc=12
    m.param('par6').set('Nt','1200');     % 物理带材数/饼 (10K)
    m.param('par6').set('Ip','583');      % 单带电流 (10K)
    m.param('par6').set('T','10[K]');     % 温度
    % 三段式不动: Nt1=30, Nt2=70, nt1/nt2/nt3=1/10/50; Nt3=Nt-2Nt1-2Nt2 自动
    Nt1=m.param.evaluate('Nt1'); Nt2=m.param.evaluate('Nt2'); Nt3=m.param.evaluate('Nt3');
    nt1=m.param.evaluate('nt1'); nt2=m.param.evaluate('nt2'); nt3=m.param.evaluate('nt3');
    Np=m.param.evaluate('Np');
    e1=Nt1/nt1; e2=Nt2/nt2; e3=Nt3/nt3; perpan=e1*2+e2*2+e3;
    fprintf('NEW: Np=12 Nt=1200 Ip=583 T=10K | Nt1=%g Nt2=%g Nt3=%g | nt=%g/%g/%g\n',Nt1,Nt2,Nt3,nt1,nt2,nt3);
    fprintf('各段等效匝: 边%g×2 + 过渡%g×2 + 中%g = %g/饼 (期望94); 整除检查 Np/2=%g Nt3/nt3=%g\n',e1,e2,e3,perpan,Np/2,e3);
    % --- rebuild 几何 (catch OpArray crash) ---
    fprintf('GEOM.RUN... %s\n', datestr(now,31)); tg=tic;
    try
      m.geom('geom1').run; fprintf('GEOM_OK %.0fs, 域=%d 边界=%d\n', toc(tg), m.geom('geom1').getNDomains, m.geom('geom1').getNBoundaries);
    catch ge
      fprintf(2,'GEOM_CRASH: %s\n', ge.message); diary off; return;
    end
    % --- 数 uni2 (带材线) 实体 ---
    try
      s=mphgetselection(m.component('comp1').selection('uni2')); nb=numel(s.entities);
      fprintf('uni2(带材线)边界数=%d, per-pancake=%.1f (期望94)\n', nb, nb/(Np/2));
    catch se; fprintf('(uni2计数失败: %s)\n', se.message); end
    % --- mesh ---
    fprintf('MESH.RUN... %s\n', datestr(now,31)); tm=tic;
    try; m.mesh('mesh1').run; fprintf('MESH_OK %.0fs, 单元=%d\n', toc(tm), m.mesh('mesh1').getNumElem); catch me; fprintf(2,'MESH_ERR: %s\n', me.message); end
    mphsave(m,[base 'ARC_TA_loss_Nc12_step1.mph']);
    fprintf('SAVED ARC_TA_loss_Nc12_step1.mph\nSTEP1_OK %s\n', datestr(now,31));
  catch ME
    fprintf(2,'STEP1_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end

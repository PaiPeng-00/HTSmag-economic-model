function verify_extreme()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'verify_extreme.log']);
  try
    m=mphload([base 'ARC_TA_loss_Nc12_R2.mph']); fprintf('LOADED\n');
    % 极端点: 4.2K, Nt=1000, Ip=700, Npw=200
    m.param('par6').set('T','4.2[K]'); m.param('par6').set('Nt','1000'); m.param('par6').set('Ip','700');
    m.param('par2').set('Npw','200'); m.param('par7').set('rho_turn','5000[uohm*cm^2]');
    m.study('std1').feature('param').active(false); m.study('std1').feature('param1').active(false);
    m.study('std1').feature('time').set('tlist','range(0,td/8,td)');
    t1=m.sol('sol1').feature('t1');
    t1.set('tlist','range(0,td/8,td)'); t1.set('rtol','1e-4'); t1.set('initialstepbdfactive',false);
    t1.set('maxstepbdfactive',true); t1.set('maxstepbdf','4000');
    m.mesh('mesh1').feature('dis1').set('numelem',20); m.mesh('mesh1').feature('size2').set('hmax',0.005);
    % 几何重建 (Nt=1000)
    fprintf('GEOM.RUN (Nt=1000)... %s\n', datestr(now,31)); m.geom('geom1').run;
    fprintf('GEOM_OK 线圈R=[%.2f,%.2f]\n', m.param.evaluate('Rin'), m.param.evaluate('Rin')+m.param.evaluate('R2'));
    e=1e-3;
    pmc=mphselectbox(m,'geom1',[3-e 6+e; -e e],'boundary');
    mi=unique([mphselectbox(m,'geom1',[3-e 3+e; -e 1.2+e],'boundary') mphselectbox(m,'geom1',[6-e 6+e; -e 1.2+e],'boundary') mphselectbox(m,'geom1',[3-e 6+e; 1.2-e 1.2+e],'boundary')]);
    m.physics('mf').feature('pmc1').selection.set(pmc); m.physics('mf').feature('mi2').selection.set(mi);
    m.mesh('mesh1').run;
    nb=numel(mphgetselection(m.component('comp1').selection('uni2')).entities);
    fprintf('带材线=%d, per-pancake=%.1f (Nt=1000期望90), 单元=%d, PMC=%s\n', nb, nb/6, m.mesh('mesh1').getNumElem, mat2str(pmc));
    fprintf('SOLVING (4.2K,Npw=200)... %s\n', datestr(now,31)); ts=tic; m.sol('sol1').runAll; el=toc(ts); fprintf('SOLVED %.1fmin\n', el/60);
    try; m.result.table.create('tmag','Table'); catch; end
    m.result.numerical('int4').set('table','tmag'); m.result.numerical('int4').set('data','dset1'); m.result.numerical('int4').setResult; Dm=mphtable(m,'tmag').data;
    try; m.result.table.create('trad','Table'); catch; end
    m.result.numerical('gev9').set('table','trad'); m.result.numerical('gev9').set('data','dset1'); m.result.numerical('gev9').setResult; Dr=mphtable(m,'trad').data;
    fprintf('=== 极端点 4.2K/Npw=200 ===\n');
    fprintf('磁化 int4: '); fprintf('%.3f ', Dm(:,2)); fprintf('-> peak=%.3f W (应光滑无伪峰)\n', max(Dm(:,2)));
    fprintf('径向 gev9: '); fprintf('%.1f ', Dr(:,2)); fprintf('-> peak=%.1f W\n', max(Dr(:,2)));
    writematrix([Dm(:,1) Dm(:,2) Dr(:,2)], [res 'ARC_sweep_T4.2_Npw200.csv']);
    fprintf('存 ARC_sweep_T4.2_Npw200.csv (扫描可复用)\nVERIFY_EXTREME_OK\n');
  catch ME
    fprintf(2,'VERIFY_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end

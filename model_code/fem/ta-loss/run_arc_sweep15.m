function run_arc_sweep15()
  % ARC 15点扫描: 3温度(各自Nt/Ip) × 5 Npw. 优化求解器(rtol=1e-4+maxstep=4000s+网格粗化).
  % 断点续跑(跳过已有CSV). 每点导出 [t, int4(磁化), gev9(径向)].
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'sweep15.log']);
  fprintf('=== ARC 15点扫描 START %s ===\n', datestr(now,31));
  % 温度 -> [Nt, Ip] (FEM 权威, 载流比0.7)
  Tlist=[4.2 10 20]; Nt_of=containers.Map([4.2 10 20],[1000 1200 1750]); Ip_of=containers.Map([4.2 10 20],[700 583 400]);
  Npw_list=[5 10 20 50 100 200]; rho=5000;
  m=mphload([base 'ARC_TA_loss_Nc12_R2.mph']); fprintf('LOADED\n');
  m.study('std1').feature('param').active(false); m.study('std1').feature('param1').active(false);
  m.study('std1').feature('time').set('tlist','range(0,td/8,td)');
  t1=m.sol('sol1').feature('t1');
  t1.set('tlist','range(0,td/8,td)'); t1.set('rtol','1e-4'); t1.set('initialstepbdfactive',false);
  t1.set('maxstepbdfactive',true); t1.set('maxstepbdf','4000');
  m.mesh('mesh1').feature('dis1').set('numelem',20); m.mesh('mesh1').feature('size2').set('hmax',0.005);
  m.param('par7').set('rho_turn',[num2str(rho) '[uohm*cm^2]']);
  n=0; ntot=numel(Tlist)*numel(Npw_list);
  for T=Tlist
    % 外层: 设温度对应的 Nt/Ip, 重建几何, 重设PMC/MI, 重网格
    m.param('par6').set('T',[num2str(T) '[K]']); m.param('par6').set('Nt',num2str(Nt_of(T))); m.param('par6').set('Ip',num2str(Ip_of(T)));
    fprintf('\n--- T=%gK: Nt=%d, Ip=%d, 重建几何... %s ---\n', T, Nt_of(T), Ip_of(T), datestr(now,31));
    m.geom('geom1').run;
    e=1e-3;
    pmc=mphselectbox(m,'geom1',[3-e 6+e; -e e],'boundary');
    mi=unique([mphselectbox(m,'geom1',[3-e 3+e; -e 1.2+e],'boundary') mphselectbox(m,'geom1',[6-e 6+e; -e 1.2+e],'boundary') mphselectbox(m,'geom1',[3-e 6+e; 1.2-e 1.2+e],'boundary')]);
    m.physics('mf').feature('pmc1').selection.set(pmc); m.physics('mf').feature('mi2').selection.set(mi);
    m.mesh('mesh1').run;
    nb=numel(mphgetselection(m.component('comp1').selection('uni2')).entities);
    fprintf('  几何OK: 带材线=%d(%.0f/饼), 单元=%d, PMC=%s\n', nb, nb/(Nt_of(T)*0+6), m.mesh('mesh1').getNumElem, mat2str(pmc));
    for Npw=Npw_list
      n=n+1; csv=[res sprintf('ARC_sweep_T%g_Npw%d.csv',T,Npw)];
      if exist(csv,'file'); fprintf('[%d/%d] SKIP T=%g Npw=%d (已有)\n',n,ntot,T,Npw); continue; end
      m.param('par2').set('Npw',num2str(Npw));
      fprintf('[%d/%d] SOLVING T=%g Npw=%d ... %s\n',n,ntot,T,Npw,datestr(now,31)); ts=tic;
      try
        m.sol('sol1').runAll;
        try; m.result.table.create('tmag','Table'); catch; end
        m.result.numerical('int4').set('table','tmag'); m.result.numerical('int4').set('data','dset1'); m.result.numerical('int4').setResult; Dm=mphtable(m,'tmag').data;
        try; m.result.table.create('trad','Table'); catch; end
        m.result.numerical('gev9').set('table','trad'); m.result.numerical('gev9').set('data','dset1'); m.result.numerical('gev9').setResult; Dr=mphtable(m,'trad').data;
        writematrix([Dm(:,1) Dm(:,2) Dr(:,2)], csv);
        fprintf('   OK %.1fmin  磁化peak=%.2f W, 径向peak=%.1f W\n', toc(ts)/60, max(Dm(:,2)), max(Dr(:,2)));
      catch ME
        fprintf(2,'   FAIL T=%g Npw=%d: %s\n',T,Npw,ME.message);
      end
    end
  end
  fprintf('\n=== ARC 15点扫描 DONE %s ===\nSWEEP15_COMPLETE\n', datestr(now,31));
  diary off;
end

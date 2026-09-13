function run_arc_rho_sweep()
  % ARC 补 rho 扫描 (Fig2 面板c): T=20K, Npw=20, rho={50,100,1500} (5000 已有).
  % 照 run_arc_sweep15.m 的优化求解器(rtol=1e-4 + maxstep=4000 + 粗网格 dis1=20/hmax=0.005).
  % 几何只建一次(单温度,只改 rho=par7 物理参数,不 remesh). 导出 [t, int4(mag), gev9(radial)].
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'rho_sweep.log']);
  fprintf('=== ARC rho补扫 START %s ===\n', datestr(now,31));
  T=20; Nt=1750; Ip=400; Npw=20; rho_list=[50 100 1500];
  m=mphload([base 'ARC_TA_loss_Nc12_R2.mph']); fprintf('LOADED\n');
  % 关掉参数扫描, 用优化瞬态设置(与 run_arc_sweep15 一致)
  m.study('std1').feature('param').active(false); m.study('std1').feature('param1').active(false);
  m.study('std1').feature('time').set('tlist','range(0,td/8,td)');
  t1=m.sol('sol1').feature('t1');
  t1.set('tlist','range(0,td/8,td)'); t1.set('rtol','1e-4'); t1.set('initialstepbdfactive',false);
  t1.set('maxstepbdfactive',true); t1.set('maxstepbdf','4000');
  m.mesh('mesh1').feature('dis1').set('numelem',20); m.mesh('mesh1').feature('size2').set('hmax',0.005);
  % T=20K 的 Nt/Ip, 建几何 + 重设 PMC/MI + 网格 (一次)
  m.param('par6').set('T',[num2str(T) '[K]']); m.param('par6').set('Nt',num2str(Nt)); m.param('par6').set('Ip',num2str(Ip));
  m.param('par2').set('Npw',num2str(Npw));
  fprintf('build geom T=20K Nt=1750 Ip=400 Npw=20 ... %s\n', datestr(now,31));
  m.geom('geom1').run;
  e=1e-3;
  pmc=mphselectbox(m,'geom1',[3-e 6+e; -e e],'boundary');
  mi=unique([mphselectbox(m,'geom1',[3-e 3+e; -e 1.2+e],'boundary') mphselectbox(m,'geom1',[6-e 6+e; -e 1.2+e],'boundary') mphselectbox(m,'geom1',[3-e 6+e; 1.2-e 1.2+e],'boundary')]);
  m.physics('mf').feature('pmc1').selection.set(pmc); m.physics('mf').feature('mi2').selection.set(mi);
  m.mesh('mesh1').run; fprintf('GEOM+MESH OK, elems=%d %s\n', m.mesh('mesh1').getNumElem, datestr(now,31));
  for rho=rho_list
    csv=[res sprintf('ARC_sweep_T20_Npw20_rho%d.csv',rho)];
    if exist(csv,'file'); fprintf('SKIP rho=%d (已有)\n',rho); continue; end
    m.param('par7').set('rho_turn',[num2str(rho) '[uohm*cm^2]']);
    fprintf('SOLVING rho=%d ... %s\n',rho,datestr(now,31)); ts=tic;
    try
      m.sol('sol1').runAll;
      try; m.result.table.create('tmag','Table'); catch; end
      m.result.numerical('int4').set('table','tmag'); m.result.numerical('int4').set('data','dset1'); m.result.numerical('int4').setResult; Dm=mphtable(m,'tmag').data;
      try; m.result.table.create('trad','Table'); catch; end
      m.result.numerical('gev9').set('table','trad'); m.result.numerical('gev9').set('data','dset1'); m.result.numerical('gev9').setResult; Dr=mphtable(m,'trad').data;
      writematrix([Dm(:,1) Dm(:,2) Dr(:,2)], csv);
      fprintf('   OK %.1fmin  磁化peak=%.2f W, 径向peak=%.1f W\n', toc(ts)/60, max(Dm(:,2)), max(Dr(:,2)));
    catch ME
      fprintf(2,'   FAIL rho=%d: %s\n',rho,ME.message);
    end
  end
  fprintf('\n=== rho补扫 DONE %s ===\nRHO_SWEEP_COMPLETE\n', datestr(now,31));
  diary off;
end
